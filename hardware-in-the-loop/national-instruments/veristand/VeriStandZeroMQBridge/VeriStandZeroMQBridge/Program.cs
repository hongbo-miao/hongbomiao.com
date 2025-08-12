using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Google.Protobuf;
using NationalInstruments.VeriStand.ClientAPI;
using NetMQ;
using NetMQ.Sockets;
using VeriStandZeroMQBridge;

public class Configuration
{
    public string VeristandGatewayHost { get; set; }
    public string SystemDefinitionPath { get; set; }
    public uint VeriStandConnectionTimeoutMs { get; set; }
    public int ZeroMQPort { get; set; }
    public int ProducerNumber { get; set; }
    public int TargetFrequencyHz { get; set; }
    public int CalibrationTimeS { get; set; }

    public static Configuration Load()
    {
        string env = Environment.GetEnvironmentVariable("APP_ENVIRONMENT") ?? "development";
        string envFile = $".env.{env.ToLower()}";

        if (!File.Exists(envFile))
        {
            throw new FileNotFoundException($"Environment file not found: {envFile}");
        }

        DotNetEnv.Env.Load(envFile);

        return new Configuration
        {
            VeristandGatewayHost = Environment.GetEnvironmentVariable("VERISTAND_GATEWAY_HOST"),
            SystemDefinitionPath = Environment.GetEnvironmentVariable("SYSTEM_DEFINITION_PATH"),
            VeriStandConnectionTimeoutMs = uint.Parse(
                Environment.GetEnvironmentVariable("VERISTAND_CONNECTION_TIMEOUT_MS")
            ),
            ZeroMQPort = int.Parse(Environment.GetEnvironmentVariable("ZEROMQ_PORT")),
            ProducerNumber = int.Parse(Environment.GetEnvironmentVariable("PRODUCER_NUMBER")),
            TargetFrequencyHz = int.Parse(
                Environment.GetEnvironmentVariable("TARGET_FREQUENCY_HZ")
            ),
            CalibrationTimeS = int.Parse(Environment.GetEnvironmentVariable("CALIBRATION_TIME_S")),
        };
    }
}

public record DataQualityMetrics
{
    public int TotalReadAttemptCount { get; set; }
    public int FailedReadCount { get; set; }
    public int LastGoodValueUsageCount { get; set; }
    public DateTime LastGoodValueTimestamp { get; set; }
    public TimeSpan LongestLastGoodValuePeriod { get; set; }
    public double LastGoodValuePercentage =>
        TotalReadAttemptCount > 0 ? (LastGoodValueUsageCount * 100.0 / TotalReadAttemptCount) : 0;

    public override string ToString()
    {
        return $"Total Reads: {TotalReadAttemptCount}, "
            + $"Failed: {FailedReadCount}, "
            + $"Last Good Value Usage: {LastGoodValuePercentage:F2}%, "
            + $"Longest Last Good Value Period: {LongestLastGoodValuePeriod.TotalMilliseconds:F0}ms";
    }
}

public static class StringSanitizer
{
    private static (int leading, int trailing) CountLeadingTrailingUnderscores(string s)
    {
        int leadingCount = s.Length - s.TrimStart('_').Length;
        int trailingCount = s.Length - s.TrimEnd('_').Length;
        return (leadingCount, trailingCount);
    }

    public static string SanitizeSignalName(string s)
    {
        if (string.IsNullOrEmpty(s))
            return s;

        (int leadingUnderscores, int trailingUnderscores) = CountLeadingTrailingUnderscores(s);

        // Replace special characters
        s = s.Replace("Δ", "d")
            .Replace("²", "2")
            .Replace("³", "3")
            .Replace("°", "deg")
            .Replace("℃", "degc")
            .Replace("%", "pct")
            .Replace("Ω", "ohm")
            .Replace("µ", "u")
            .Replace("/", "."); // Convert forward slashes to dots

        // Replace special characters with underscore, except for dots
        s = Regex.Replace(s, @"[^a-zA-Z0-9_\.]", "_");

        // Replace multiple underscores with single underscore
        s = Regex.Replace(s, @"_+", "_");

        // Remove leading and trailing underscores
        s = s.Trim('_');

        // Add back original leading and trailing underscores
        s = new string('_', leadingUnderscores) + s + new string('_', trailingUnderscores);

        return s;
    }
}

public class Program
{
    private static Configuration config;
    private static Channel<byte[]> channel;
    private static volatile bool isRunning = true;
    private static int messageCount = 0;
    private static int messagesLastInterval = 0;
    private static readonly DataQualityMetrics metrics = new DataQualityMetrics();
    private static long nextGlobalTimestampNs =
        DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000;
    private static readonly object timestampLock = new object();
    private static SystemState lastKnownState;

    private static bool GetIsVeriStandActive(IWorkspace2 workspace)
    {
        SystemState currentState;
        string systemDefinitionFile;
        string[] targets;
        workspace.GetSystemState(out currentState, out systemDefinitionFile, out targets);

        // Only log if the state has changed
        if (currentState != lastKnownState)
        {
            Console.WriteLine(
                $"[VeriStand State] State changed from {lastKnownState} to {currentState}"
            );
            lastKnownState = currentState;
        }

        return currentState.ToString() == "Active";
    }

    private static long GetNextTimestamp()
    {
        lock (timestampLock)
        {
            long timestampNs = nextGlobalTimestampNs;
            nextGlobalTimestampNs += (1000000 / config.TargetFrequencyHz);
            return timestampNs * 1000;
        }
    }

    public static async Task Main(string[] args)
    {
        // Add console cancel handler
        Console.CancelKeyPress += (sender, e) =>
        {
            e.Cancel = true; // Prevent immediate termination
            isRunning = false;
            Console.WriteLine("\n[Shutdown] Graceful shutdown initiated...");
        };

        try
        {
            config = Configuration.Load();
            await Console.Out.WriteLineAsync(
                $"[Config] Loaded configuration for environment: {Environment.GetEnvironmentVariable("APP_ENVIRONMENT") ?? "development"}"
            );

            channel = Channel.CreateUnbounded<byte[]>(
                new UnboundedChannelOptions { SingleReader = true, SingleWriter = false }
            );

            using (PublisherSocket publisher = new PublisherSocket())
            {
                string zeroMQUrl = $"tcp://*:{config.ZeroMQPort}";
                publisher.Bind(zeroMQUrl);
                await Console.Out.WriteLineAsync($"[ZMQ] Publisher bound to {zeroMQUrl}");

                try
                {
                    IWorkspace2 workspace = new Factory().GetIWorkspace2(
                        config.VeristandGatewayHost
                    );

                    workspace.ConnectToSystem(
                        config.SystemDefinitionPath,
                        true,
                        config.VeriStandConnectionTimeoutMs
                    );

                    if (!GetIsVeriStandActive(workspace))
                    {
                        throw new Exception("VeriStand is not active. Cannot start streaming.");
                    }

                    await Console.Out.WriteLineAsync("[Status] Data collection started");

                    string[] aliases,
                        channels;
                    workspace.GetAliasList(out aliases, out channels);
                    await Console.Out.WriteLineAsync(
                        $"[Config] Aliases Count: {aliases.Length} | Aliases: {string.Join(", ", aliases)}"
                    );
                    await Console.Out.WriteLineAsync(
                        $"[Config] Channels Count: {channels.Length} | Channels: {string.Join(", ", channels)}"
                    );

                    double[] calibrationValues = new double[channels.Length];
                    double[] calibrationLastGoodValues = new double[channels.Length];
                    var calibrationResults = await PerformCalibration(
                        workspace,
                        channels,
                        calibrationValues,
                        calibrationLastGoodValues
                    );

                    List<Task> producerTasks = new List<Task>();
                    for (int i = 0; i < config.ProducerNumber; i++)
                    {
                        producerTasks.Add(
                            Task.Run(() => ProducerTask(workspace, channels, calibrationResults))
                        );
                    }

                    Task consumerTask = Task.Run(async () =>
                    {
                        await foreach (byte[] data in channel.Reader.ReadAllAsync())
                        {
                            if (!isRunning)
                                break;

                            // Signals signals = Signals.Parser.ParseFrom(data);
                            // DateTime datetime = DateTimeOffset
                            //     .FromUnixTimeMilliseconds(signals.TimestampNs / 1000000)
                            //     .DateTime;
                            // Console.WriteLine(
                            //     $"Timestamp: {signals.TimestampNs} ({datetime:HH:mm:ss.fff})"
                            // );
                            // string signalValues = string.Join(
                            //     " | ",
                            //     signals.Signals_.Select(s => $"{s.Name}: {s.Value:F3}")
                            // );
                            // Console.WriteLine($"Signal values: {signalValues}");

                            publisher.SendFrame(data);
                            Interlocked.Increment(ref messageCount);
                            Interlocked.Increment(ref messagesLastInterval);
                        }
                    });

                    Task monitoringTask = Task.Run(MonitorFrequency);

                    await Task.WhenAll(
                        producerTasks.Concat(new[] { consumerTask, monitoringTask })
                    );
                }
                catch (Exception ex)
                {
                    await Console.Out.WriteLineAsync($"[ERROR] {ex.Message}");
                    Environment.Exit(1);
                }
            }
        }
        catch (Exception ex)
        {
            await Console.Out.WriteLineAsync($"[ERROR] Configuration error: {ex.Message}");
            Environment.Exit(1);
        }
    }

    private static Signals CreateSignalsMessage(
        string[] channels,
        double[] values,
        bool isUsingLastGoodValues,
        double[] lastGoodValues
    )
    {
        return new Signals
        {
            TimestampNs = GetNextTimestamp(),
            Signals_ =
            {
                channels.Zip(
                    isUsingLastGoodValues ? lastGoodValues : values,
                    (name, value) =>
                        new Signal
                        {
                            Name = StringSanitizer.SanitizeSignalName(name),
                            Value = (float)value,
                        }
                ),
            },
            SkippedTickNumber = 0,
            IsUsingLastGoodValues = isUsingLastGoodValues,
            FrequencyHz = config.TargetFrequencyHz,
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
        long calibrationEndTime = config.CalibrationTimeS * Stopwatch.Frequency;

        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Start();
        long periodTicks = Stopwatch.Frequency / config.TargetFrequencyHz;
        long nextTick = stopwatch.ElapsedTicks;

        Console.WriteLine("Starting calibration...");

        while (calibrationStopwatch.ElapsedTicks < calibrationEndTime && isRunning)
        {
            if (!GetIsVeriStandActive(workspace))
            {
                throw new Exception("VeriStand became inactive during calibration.");
            }

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
        int readEveryNSampleCount = Math.Max(1, config.TargetFrequencyHz / actualReadFrequency);

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

        return (readEveryNSampleCount, Stopwatch.Frequency / config.TargetFrequencyHz);
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

        int stateCheckCounter = 0;
        const int STATE_CHECK_INTERVAL = 1000; // Check state every 1000 iterations
        const int RECOVERY_WAIT_MS = 1000; // Wait 1 second before trying to recover

        while (isRunning)
        {
            if (++stateCheckCounter >= STATE_CHECK_INTERVAL)
            {
                if (!GetIsVeriStandActive(workspace))
                {
                    await Console.Out.WriteLineAsync(
                        "[Producer] VeriStand is not active. Waiting for recovery..."
                    );

                    // Wait for VeriStand to become active again
                    while (isRunning)
                    {
                        await Task.Delay(RECOVERY_WAIT_MS);
                        if (GetIsVeriStandActive(workspace))
                        {
                            await Console.Out.WriteLineAsync(
                                "[Producer] VeriStand is active again. Resuming stream..."
                            );
                            // Reset the stopwatch and nextTick to ensure proper timing after recovery
                            stopwatch.Restart();
                            nextTick = stopwatch.ElapsedTicks;
                            break;
                        }
                    }
                }
                stateCheckCounter = 0;
            }

            if (stopwatch.ElapsedTicks >= nextTick)
            {
                bool isUsingLastGoodValues = false;
                try
                {
                    if (metrics.TotalReadAttemptCount % readEveryNSampleCount == 0)
                    {
                        workspace.GetMultipleChannelValues(channels, out values);
                        Array.Copy(values, lastGoodValues, values.Length);
                        metrics.LastGoodValueTimestamp = DateTime.UtcNow;
                    }
                    else
                    {
                        isUsingLastGoodValues = true;
                    }

                    UpdateMetrics(isUsingLastGoodValues);

                    Signals signals = CreateSignalsMessage(
                        channels,
                        values,
                        isUsingLastGoodValues,
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
                    metrics.FailedReadCount++;
                    await Task.Delay(10);
                }
            }
        }
    }

    private static void UpdateMetrics(bool isUsingLastGoodValues)
    {
        metrics.TotalReadAttemptCount++;
        if (isUsingLastGoodValues)
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

            if (lastKnownState.ToString() == "Active")
            {
                long currentTime = stopwatch.ElapsedMilliseconds;
                int currentCount = messageCount;
                int messages = currentCount - lastCount;
                double actualFrequency = messages / ((currentTime - lastTime) / 1000.0);

                Console.WriteLine($"Actual frequency: {actualFrequency:F2} Hz");
                Console.WriteLine($"Target frequency: {config.TargetFrequencyHz} Hz");
                Console.WriteLine(
                    $"Difference: {actualFrequency - config.TargetFrequencyHz:F2} Hz"
                );
                Console.WriteLine($"Data Quality: {metrics}");

                lastCount = currentCount;
                lastTime = currentTime;
            }
        }
    }
}
