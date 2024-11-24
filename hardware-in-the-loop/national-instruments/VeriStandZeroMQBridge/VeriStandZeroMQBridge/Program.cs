using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Google.Protobuf;
using NationalInstruments.VeriStand.ClientAPI;
using NetMQ;
using NetMQ.Sockets;
using VeriStandZeroMQBridge;

public record DataQualityMetrics
{
    public int TotalReadAttempts { get; set; }
    public int FailedReads { get; set; }
    public int LastGoodValueUsageCount { get; set; }
    public DateTime LastGoodValueTimestamp { get; set; }
    public TimeSpan LongestLastGoodValuePeriod { get; set; }
    public double LastGoodValuePercentage =>
        TotalReadAttempts > 0 ? (LastGoodValueUsageCount * 100.0 / TotalReadAttempts) : 0;

    public override string ToString()
    {
        return $"Total Reads: {TotalReadAttempts}, "
            + $"Failed: {FailedReads}, "
            + $"Last Good Value Usage: {LastGoodValuePercentage:F2}%, "
            + $"Longest Last Good Value Period: {LongestLastGoodValuePeriod.TotalMilliseconds:F0}ms";
    }
}

public class Program
{
    private const string GATEWAY_IP = "localhost";
    private const string SYSTEM_DEFINITION_PATH =
        @"C:\Users\Public\Documents\National Instruments\NI VeriStand 2024\Examples\Stimulus Profile\Engine Demo\Engine Demo.nivssdf";
    private const int VERISTAND_CONNECTION_TIMEOUT_MS = 60000;
    private const string ZEROMQ_ADDRESS = "tcp://*:5555";
    private const int PRODUCER_NUMBER = 4;
    private const int TARGET_FREQUENCY_HZ = 100;
    private const int CALIBRATION_TIME_S = 2;

    private static Channel<byte[]> channel;
    private static volatile bool isRunning = true;
    private static int messageCount = 0;
    private static int messagesLastInterval = 0;
    private static readonly DataQualityMetrics metrics = new DataQualityMetrics();
    private static long nextGlobalTimestampNs =
        DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000;
    private static readonly object timestampLock = new object();

    private static string SanitizeSignalName(string name)
    {
        return new string(name.Where(c => char.IsLetterOrDigit(c)).ToArray());
    }

    private static long GetNextTimestamp()
    {
        lock (timestampLock)
        {
            long timestampNs = nextGlobalTimestampNs;
            nextGlobalTimestampNs += (1000000 / TARGET_FREQUENCY_HZ);
            return timestampNs * 1000; // Convert to nanoseconds
        }
    }

    public static async Task Main(string[] args)
    {
        channel = Channel.CreateUnbounded<byte[]>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = false }
        );

        using (PublisherSocket publisher = new PublisherSocket())
        {
            publisher.Bind(ZEROMQ_ADDRESS);
            await Console.Out.WriteLineAsync($"[ZeroMQ] Publisher bound to {ZEROMQ_ADDRESS}");

            try
            {
                IWorkspace2 workspace = new Factory().GetIWorkspace2(GATEWAY_IP);
                string[] aliases,
                    channels;
                workspace.GetAliasList(out aliases, out channels);
                await Console.Out.WriteLineAsync(
                    $"[Config] Aliases Count: {aliases.Length} | Aliases: {string.Join(", ", aliases)}"
                );
                await Console.Out.WriteLineAsync(
                    $"[Config] Channels Count: {channels.Length} | Channels: {string.Join(", ", channels)}"
                );

                workspace.ConnectToSystem(
                    SYSTEM_DEFINITION_PATH,
                    true,
                    VERISTAND_CONNECTION_TIMEOUT_MS
                );
                await Console.Out.WriteLineAsync("[Status] Data collection started");

                // Perform calibration once before starting producer tasks
                double[] calibrationValues = new double[channels.Length];
                double[] calibrationLastGoodValues = new double[channels.Length];
                var calibrationResults = await PerformCalibration(
                    workspace,
                    channels,
                    calibrationValues,
                    calibrationLastGoodValues
                );

                Stopwatch totalStopwatch = Stopwatch.StartNew();

                List<Task> producerTasks = new List<Task>();
                for (int i = 0; i < PRODUCER_NUMBER; i++)
                {
                    producerTasks.Add(
                        Task.Run(() => ProducerTask(workspace, channels, calibrationResults))
                    );
                }

                Task consumerTask = Task.Run(async () =>
                {
                    await foreach (byte[] data in channel.Reader.ReadAllAsync())
                    {
                        // var signals = Signals.Parser.ParseFrom(data);
                        // var datetime = DateTimeOffset
                        //     .FromUnixTimeMilliseconds(signals.TimestampNs / 1000000)
                        //     .DateTime;
                        // Console.WriteLine($"Timestamp: {signals.TimestampNs} ({datetime})");

                        publisher.SendFrame(data);
                        Interlocked.Increment(ref messageCount);
                        Interlocked.Increment(ref messagesLastInterval);
                    }
                });

                Task monitoringTask = Task.Run(MonitorFrequency);

                await Task.WhenAll(producerTasks.Concat(new[] { consumerTask, monitoringTask }));

                double totalTime = totalStopwatch.Elapsed.TotalSeconds;
                double averageMessagesPerSecond = messageCount / totalTime;
                await Console.Out.WriteLineAsync(
                    $"[Complete] Runtime: {totalTime:F2}s | Avg Speed: {averageMessagesPerSecond:F2} msg/s | Total Messages: {messageCount:N0}"
                );
            }
            catch (Exception ex)
            {
                await Console.Out.WriteLineAsync($"[ERROR] {ex.Message}");
                Environment.Exit(1);
            }
        }
    }

    private static Signals CreateSignalsMessage(
        string[] channels,
        double[] values,
        bool useLastGoodValues,
        double[] lastGoodValues
    )
    {
        return new Signals
        {
            TimestampNs = GetNextTimestamp(),
            Signals_ =
            {
                channels.Zip(
                    useLastGoodValues ? lastGoodValues : values,
                    (name, value) =>
                        new Signal
                        {
                            Name = SanitizeSignalName(name),
                            Value = (float)value,
                            IsLastGoodValue = useLastGoodValues,
                        }
                ),
            },
            SkippedTickNumber = 0,
            IsUsingLastGoodValues = useLastGoodValues,
            FrequencyHz = TARGET_FREQUENCY_HZ,
        };
    }

    private static async Task<(int readEveryNSampleCount, long periodTicks)> PerformCalibration(
        IWorkspace2 workspace,
        string[] channels,
        double[] values,
        double[] lastGoodValues
    )
    {
        List<long> sampleTimes = new List<long>();
        Stopwatch calibrationStopwatch = Stopwatch.StartNew();
        long calibrationEndTime = CALIBRATION_TIME_S * Stopwatch.Frequency;

        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Start();
        long periodTicks = Stopwatch.Frequency / TARGET_FREQUENCY_HZ;
        long nextTick = stopwatch.ElapsedTicks;

        Console.WriteLine("Starting calibration...");

        while (calibrationStopwatch.ElapsedTicks < calibrationEndTime && isRunning)
        {
            if (stopwatch.ElapsedTicks >= nextTick)
            {
                long beforeRead = calibrationStopwatch.ElapsedTicks;

                try
                {
                    workspace.GetMultipleChannelValues(channels, out values);
                    long readTime = calibrationStopwatch.ElapsedTicks - beforeRead;
                    sampleTimes.Add(readTime);
                    Array.Copy(values, lastGoodValues, values.Length);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Calibration error: {ex.Message}");
                }

                nextTick += periodTicks;
                HandlePreciseTiming(nextTick);
            }
        }

        return CalculateCalibrationResults(sampleTimes);
    }

    private static (int readEveryNSampleCount, long periodTicks) CalculateCalibrationResults(
        List<long> sampleTimes
    )
    {
        long averageReadTime = (long)sampleTimes.Average();
        long minReadTime = sampleTimes.Min();
        long maxReadTime = sampleTimes.Max();
        double stdDev = Math.Sqrt(sampleTimes.Average(v => Math.Pow(v - averageReadTime, 2)));
        int sampleCount = sampleTimes.Count;

        long[] sortedTimes = sampleTimes.OrderBy(x => x).ToArray();
        long percentile95 = sortedTimes[(int)(sortedTimes.Length * 0.95)];
        long minReadPeriodTicks = percentile95 * 2;
        int actualReadFrequency = (int)(Stopwatch.Frequency / minReadPeriodTicks);
        int readEveryNSampleCount = Math.Max(1, TARGET_FREQUENCY_HZ / actualReadFrequency);

        Console.WriteLine("\n========== CALIBRATION RESULTS ==========");
        Console.WriteLine($"Samples collected: {sampleCount}");
        Console.WriteLine(
            $"Average read time: {averageReadTime * 1000.0 / Stopwatch.Frequency:F3}ms"
        );
        Console.WriteLine($"Min read time: {minReadTime * 1000.0 / Stopwatch.Frequency:F3}ms");
        Console.WriteLine($"Max read time: {maxReadTime * 1000.0 / Stopwatch.Frequency:F3}ms");
        Console.WriteLine($"Standard deviation: {stdDev * 1000.0 / Stopwatch.Frequency:F3}ms");
        Console.WriteLine($"Measured frequency: {Stopwatch.Frequency / averageReadTime:F1} Hz");
        Console.WriteLine($"Safe read frequency: {actualReadFrequency} Hz");
        Console.WriteLine($"Reading every {readEveryNSampleCount} samples");
        Console.WriteLine("========================================\n");

        return (readEveryNSampleCount, Stopwatch.Frequency / TARGET_FREQUENCY_HZ);
    }

    private static void HandlePreciseTiming(long nextTick)
    {
        long sleepTicks = nextTick - Stopwatch.GetTimestamp();
        if (sleepTicks > 0)
        {
            long sleepMillis = sleepTicks * 1000 / Stopwatch.Frequency;
            if (sleepMillis > 1)
            {
                Task.Delay(1).Wait();
            }
            SpinWait.SpinUntil(() => Stopwatch.GetTimestamp() >= nextTick);
        }
    }

    private static async Task ProducerTask(
        IWorkspace2 workspace,
        string[] channels,
        (int readEveryNSampleCount, long periodTicks) calibrationResults
    )
    {
        double[] values = new double[channels.Length];
        double[] lastGoodValues = new double[channels.Length];

        var (readEveryNSampleCount, periodTicks) = calibrationResults;

        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Start();
        long nextTick = stopwatch.ElapsedTicks;

        while (isRunning)
        {
            if (stopwatch.ElapsedTicks >= nextTick)
            {
                bool useLastGoodValues = false;
                try
                {
                    if (metrics.TotalReadAttempts % readEveryNSampleCount == 0)
                    {
                        workspace.GetMultipleChannelValues(channels, out values);
                        Array.Copy(values, lastGoodValues, values.Length);
                        metrics.LastGoodValueTimestamp = DateTime.UtcNow;
                    }
                    else
                    {
                        useLastGoodValues = true;
                    }

                    UpdateMetrics(useLastGoodValues);

                    Signals signals = CreateSignalsMessage(
                        channels,
                        values,
                        useLastGoodValues,
                        lastGoodValues
                    );
                    byte[] data = signals.ToByteArray();
                    await channel.Writer.WriteAsync(data);

                    nextTick += periodTicks;
                    HandlePreciseTiming(nextTick);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error during normal operation: {ex.Message}");
                    metrics.FailedReads++;
                    await Task.Delay(10);
                }
            }
        }
    }

    private static void UpdateMetrics(bool useLastGoodValues)
    {
        metrics.TotalReadAttempts++;
        if (useLastGoodValues)
        {
            metrics.LastGoodValueUsageCount++;
            TimeSpan currentLgvPeriod = DateTime.UtcNow - metrics.LastGoodValueTimestamp;
            if (currentLgvPeriod > metrics.LongestLastGoodValuePeriod)
            {
                metrics.LongestLastGoodValuePeriod = currentLgvPeriod;
            }
        }
    }

    private static void MonitorFrequency()
    {
        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Start();
        int lastCount = 0;
        long lastTime = 0;

        while (isRunning)
        {
            Thread.Sleep(1000);

            long currentTime = stopwatch.ElapsedMilliseconds;
            int currentCount = messageCount;
            int messages = currentCount - lastCount;
            double actualFrequency = messages / ((currentTime - lastTime) / 1000.0);

            Console.WriteLine($"Actual frequency: {actualFrequency:F2} Hz");
            Console.WriteLine($"Target frequency: {TARGET_FREQUENCY_HZ} Hz");
            Console.WriteLine($"Difference: {actualFrequency - TARGET_FREQUENCY_HZ:F2} Hz");
            Console.WriteLine($"Data Quality: {metrics}");

            lastCount = currentCount;
            lastTime = currentTime;
        }
    }
}
