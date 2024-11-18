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

public class Program
{
    private const string GATEWAY_IP = "localhost";
    private const string SYSTEM_DEFINITION_PATH =
        @"C:\Users\Public\Documents\National Instruments\NI VeriStand 2024\Examples\Stimulus Profile\Engine Demo\Engine Demo.nivssdf";
    private const int TOTAL_MESSAGES = 500000;
    private const int INTERVAL_MS = 5000;
    private const int CONNECTION_TIMEOUT_MS = 60000;
    private const string ZMQ_ADDRESS = "tcp://*:5555";
    private const int NUM_PRODUCERS = 4;
    private const int CHANNEL_CAPACITY = 10000;

    private static Channel<byte[]> channel;
    private static volatile bool isRunning = true;
    private static readonly CancellationTokenSource cts = new CancellationTokenSource();

    private class MessageCounters
    {
        public int TotalCount;
        public int IntervalCount;
    }

    public static async Task Main(string[] args)
    {
        var options = new BoundedChannelOptions(CHANNEL_CAPACITY)
        {
            FullMode = BoundedChannelFullMode.Wait,
        };
        channel = Channel.CreateBounded<byte[]>(options);

        using (var publisher = new PublisherSocket())
        {
            publisher.Bind(ZMQ_ADDRESS);
            await Console.Out.WriteLineAsync($"[ZMQ] Publisher bound to {ZMQ_ADDRESS}");

            try
            {
                var workspace = new Factory().GetIWorkspace2(GATEWAY_IP);
                string[] aliases,
                    channels;
                workspace.GetAliasList(out aliases, out channels);
                await Console.Out.WriteLineAsync(
                    $"[Config] Aliases Count: {aliases.Length} | Aliases: {string.Join(", ", aliases)}"
                );
                await Console.Out.WriteLineAsync(
                    $"[Config] Channels Count: {channels.Length} | Aliases: {string.Join(", ", channels)}"
                );

                workspace.ConnectToSystem(SYSTEM_DEFINITION_PATH, true, CONNECTION_TIMEOUT_MS);
                await Console.Out.WriteLineAsync("[Status] Data collection started");

                var totalStopwatch = Stopwatch.StartNew();
                var counters = new MessageCounters();

                // Start producer tasks
                var producerTasks = new List<Task>();
                for (int i = 0; i < NUM_PRODUCERS; i++)
                {
                    producerTasks.Add(ProducerTask(workspace, aliases));
                }

                // Consumer task
                var consumerTask = ConsumeAsync(publisher, counters);

                // Monitoring task
                while (isRunning)
                {
                    await Task.Delay(INTERVAL_MS);
                    var currentMessagesLastInterval = Interlocked.Exchange(
                        ref counters.IntervalCount,
                        0
                    );
                    double messagesPerSecond = currentMessagesLastInterval / (INTERVAL_MS / 1000.0);
                    await Console.Out.WriteLineAsync(
                        $"[Update] Speed: {messagesPerSecond:F2} msg/s | Total: {counters.TotalCount:N0}"
                    );
                }

                // Cleanup
                cts.Cancel();
                channel.Writer.Complete();
                await Task.WhenAll(producerTasks);
                await consumerTask;

                double totalTime = totalStopwatch.Elapsed.TotalSeconds;
                double averageMessagesPerSecond = counters.TotalCount / totalTime;
                await Console.Out.WriteLineAsync(
                    $"[Complete] Runtime: {totalTime:F2}s | Avg Speed: {averageMessagesPerSecond:F2} msg/s | Total Messages: {counters.TotalCount:N0}"
                );
            }
            catch (Exception ex)
            {
                await Console.Out.WriteLineAsync($"[ERROR] {ex.Message}");
                Environment.Exit(1);
            }
        }
    }

    private static async Task ProducerTask(IWorkspace2 workspace, string[] aliases)
    {
        double[] values = new double[aliases.Length];

        while (isRunning && !cts.Token.IsCancellationRequested)
        {
            workspace.GetMultipleChannelValues(aliases, out values);
            var signals = new Signals
            {
                Timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                Signals_ =
                {
                    aliases.Zip(
                        values,
                        (alias, value) => new Signal { Alias = alias, Value = value }
                    ),
                },
            };

            byte[] data = signals.ToByteArray();
            try
            {
                await channel.Writer.WriteAsync(data, cts.Token);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }
    }

    private static async Task ConsumeAsync(PublisherSocket publisher, MessageCounters counters)
    {
        await foreach (var data in channel.Reader.ReadAllAsync(cts.Token))
        {
            publisher.SendFrame(data);

            Interlocked.Increment(ref counters.TotalCount);
            Interlocked.Increment(ref counters.IntervalCount);

            if (counters.TotalCount >= TOTAL_MESSAGES)
            {
                isRunning = false;
                break;
            }
        }
    }
}
