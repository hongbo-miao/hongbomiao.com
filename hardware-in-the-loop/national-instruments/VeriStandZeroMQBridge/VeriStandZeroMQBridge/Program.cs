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

    private static Channel<byte[]> channel;
    private static volatile bool isRunning = true;
    private static int messageCount = 0;
    private static int messagesLastInterval = 0;

    public static async Task Main(string[] args)
    {
        channel = Channel.CreateUnbounded<byte[]>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = false }
        );

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
                    $"[Config] Channels Count: {channels.Length} | Channels: {string.Join(", ", channels)}"
                );

                workspace.ConnectToSystem(SYSTEM_DEFINITION_PATH, true, CONNECTION_TIMEOUT_MS);
                await Console.Out.WriteLineAsync("[Status] Data collection started");

                var totalStopwatch = Stopwatch.StartNew();

                // Start producer tasks
                var producerTasks = new List<Task>();
                for (int i = 0; i < NUM_PRODUCERS; i++)
                {
                    producerTasks.Add(Task.Run(() => ProducerTask(workspace, aliases)));
                }

                // Consumer task
                var consumerTask = Task.Run(async () =>
                {
                    await foreach (var data in channel.Reader.ReadAllAsync())
                    {
                        publisher.SendFrame(data);

                        Interlocked.Increment(ref messageCount);
                        Interlocked.Increment(ref messagesLastInterval);

                        if (messageCount >= TOTAL_MESSAGES)
                        {
                            isRunning = false;
                            break;
                        }
                    }
                });

                // Monitoring task
                while (isRunning)
                {
                    await Task.Delay(INTERVAL_MS);
                    var currentMessagesLastInterval = Interlocked.Exchange(
                        ref messagesLastInterval,
                        0
                    );
                    double messagesPerSecond = currentMessagesLastInterval / (INTERVAL_MS / 1000.0);

                    // Get current values
                    double[] currentValues = new double[channels.Length];
                    workspace.GetMultipleChannelValues(channels, out currentValues);

                    var channelValues = string.Join(
                        " | ",
                        channels.Zip(currentValues, (alias, value) => $"{alias}:{value:F2}")
                    );

                    Console.WriteLine(
                        $"[Update] Speed: {messagesPerSecond:F2} msg/s | Total: {messageCount:N0}"
                    );
                    Console.WriteLine($"[Update] Values: {channelValues}");
                }

                // Cleanup
                channel.Writer.Complete();
                await Task.WhenAll(producerTasks);
                await consumerTask;

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

    private static async Task ProducerTask(IWorkspace2 workspace, string[] aliases)
    {
        double[] values = new double[aliases.Length];

        while (isRunning)
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
            await channel.Writer.WriteAsync(data);
        }
    }
}
