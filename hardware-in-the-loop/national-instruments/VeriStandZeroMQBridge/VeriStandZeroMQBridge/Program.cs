using NetMQ;
using NetMQ.Sockets;
using NationalInstruments.VeriStand.ClientAPI;
using System;
using System.Diagnostics;
using Google.Protobuf;
using System.Linq;
using VeriStandZeroMQBridge;

public class Program
{
    private const string GATEWAY_IP = "localhost";
    private const string SYSTEM_DEFINITION_PATH = @"C:\Users\Public\Documents\National Instruments\NI VeriStand 2024\Examples\Stimulus Profile\Engine Demo\Engine Demo.nivssdf";
    private const int TOTAL_MESSAGES = 100000;
    private const int INTERVAL_MS = 5000;
    private const int CONNECTION_TIMEOUT_MS = 60000;
    private const string ZMQ_ADDRESS = "tcp://*:5555";

    public static void Main(string[] args)
    {
        using (var publisher = new PublisherSocket())
        {
            publisher.Bind(ZMQ_ADDRESS);
            Console.WriteLine($"[ZMQ] Publisher bound to {ZMQ_ADDRESS}");

            try
            {
                var workspace = new Factory().GetIWorkspace2(GATEWAY_IP);
                string[] aliases, channels;
                workspace.GetAliasList(out aliases, out channels);
                Console.WriteLine($"[Config] Aliases: {string.Join(", ", aliases)} | Channels: {string.Join(", ", channels)}");

                double[] values = new double[aliases.Length];
                workspace.ConnectToSystem(SYSTEM_DEFINITION_PATH, true, CONNECTION_TIMEOUT_MS);
                Console.WriteLine("[Status] Data collection started");

                int messageCount = 0;
                int messagesLastInterval = 0;
                var totalStopwatch = Stopwatch.StartNew();
                var intervalStopwatch = Stopwatch.StartNew();

                while (messageCount < TOTAL_MESSAGES)
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

                    // Serialize and send
                    byte[] data = signals.ToByteArray();
                    publisher.SendFrame(data);

                    messageCount++;
                    messagesLastInterval++;

                    if (intervalStopwatch.ElapsedMilliseconds >= INTERVAL_MS)
                    {
                        double messagesPerSecond = messagesLastInterval / (intervalStopwatch.ElapsedMilliseconds / 1000.0);
                        Console.WriteLine($"[Update] Speed: {messagesPerSecond:F2} msg/s | Total: {messageCount:N0} | Value Count: {values.Length}");
                        // Console.WriteLine($"Values: {string.Join(", ", values)}");

                        messagesLastInterval = 0;
                        intervalStopwatch.Restart();
                    }
                }

                double totalTime = totalStopwatch.Elapsed.TotalSeconds;
                double averageMessagesPerSecond = TOTAL_MESSAGES / totalTime;
                Console.WriteLine($"[Complete] Runtime: {totalTime:F2}s | Avg Speed: {averageMessagesPerSecond:F2} msg/s | Total Messages: {TOTAL_MESSAGES:N0}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] {ex.Message}");
                Environment.Exit(1);
            }
        }
    }
}
