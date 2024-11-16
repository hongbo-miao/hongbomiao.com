using NetMQ;
using NetMQ.Sockets;
using NationalInstruments.VeriStand.ClientAPI;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using Google.Protobuf;
using VeriStandZeroMQBridge;

public class Program
{
    private const string GATEWAY_IP = "localhost";
    private const string SYSTEM_DEFINITION_PATH = @"C:\Users\Public\Documents\National Instruments\NI VeriStand 2024\Examples\Stimulus Profile\Engine Demo\Engine Demo.nivssdf";
    private const int TOTAL_MESSAGES = 200000;
    private const int INTERVAL_MS = 5000;
    private const int CONNECTION_TIMEOUT_MS = 60000;
    private const string ZMQ_ADDRESS = "tcp://*:5555";
    private const int NUM_PRODUCERS = 4; // Number of producer threads
    private const int QUEUE_CAPACITY = 10000;

    private static BlockingCollection<byte[]> messageQueue;
    private static volatile bool isRunning = true;

    public static async Task Main(string[] args)
    {
        messageQueue = new BlockingCollection<byte[]>(QUEUE_CAPACITY);

        using (var publisher = new PublisherSocket())
        {
            publisher.Bind(ZMQ_ADDRESS);
            await Console.Out.WriteLineAsync($"[ZMQ] Publisher bound to {ZMQ_ADDRESS}");

            try
            {
                var workspace = new Factory().GetIWorkspace2(GATEWAY_IP);
                string[] aliases, channels;
                workspace.GetAliasList(out aliases, out channels);
                await Console.Out.WriteLineAsync($"[Config] Aliases Count: {aliases.Length} | Aliases: {string.Join(", ", aliases)}");
                await Console.Out.WriteLineAsync($"[Config] Channels Count: {channels.Length} | Aliases: {string.Join(", ", channels)}");

                workspace.ConnectToSystem(SYSTEM_DEFINITION_PATH, true, CONNECTION_TIMEOUT_MS);
                await Console.Out.WriteLineAsync("[Status] Data collection started");

                var totalStopwatch = Stopwatch.StartNew();
                var messageCount = 0;
                var messagesLastInterval = 0;

                // Start producer tasks
                var producerTasks = new List<Task>();
                for (int i = 0; i < NUM_PRODUCERS; i++)
                {
                    producerTasks.Add(Task.Run(() => ProducerTask(workspace, aliases)));
                }

                // Consumer task
                var consumerTask = Task.Run(() =>
                {
                    while (!messageQueue.IsCompleted)
                    {
                        try
                        {
                            var data = messageQueue.Take();
                            publisher.SendFrame(data);

                            Interlocked.Increment(ref messageCount);
                            Interlocked.Increment(ref messagesLastInterval);

                            if (messageCount >= TOTAL_MESSAGES)
                            {
                                isRunning = false;
                                break;
                            }
                        }
                        catch (InvalidOperationException)
                        {
                            break;
                        }
                    }
                });

                // Monitoring task
                while (isRunning)
                {
                    await Task.Delay(INTERVAL_MS);
                    var currentMessagesLastInterval = Interlocked.Exchange(ref messagesLastInterval, 0);
                    double messagesPerSecond = currentMessagesLastInterval / (INTERVAL_MS / 1000.0);
                    await Console.Out.WriteLineAsync($"[Update] Speed: {messagesPerSecond:F2} msg/s | Total: {messageCount:N0} | Queue Size: {messageQueue.Count}");
                }

                // Cleanup
                messageQueue.CompleteAdding();
                await Task.WhenAll(producerTasks);
                await consumerTask;

                double totalTime = totalStopwatch.Elapsed.TotalSeconds;
                double averageMessagesPerSecond = messageCount / totalTime;
                await Console.Out.WriteLineAsync($"[Complete] Runtime: {totalTime:F2}s | Avg Speed: {averageMessagesPerSecond:F2} msg/s | Total Messages: {messageCount:N0}");
            }
            catch (Exception ex)
            {
                await Console.Out.WriteLineAsync($"[ERROR] {ex.Message}");
                Environment.Exit(1);
            }
        }
    }

    private static void ProducerTask(IWorkspace2 workspace, string[] aliases)
    {
        double[] values = new double[aliases.Length];

        while (isRunning)
        {
            workspace.GetMultipleChannelValues(aliases, out values);
            var signals = new Signals
            {
                Timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                Signals_ = { aliases.Zip(values, (alias, value) =>
                    new Signal {
                        Alias = alias,
                        Value = value
                    }) }
            };

            byte[] data = signals.ToByteArray();
            if (!messageQueue.TryAdd(data))
            {
                Thread.Sleep(1); // Back off if the queue is full
            }
        }
    }
}
