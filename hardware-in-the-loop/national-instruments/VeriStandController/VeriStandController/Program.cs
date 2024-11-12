using NationalInstruments.VeriStand.ClientAPI;
using System;
using System.Diagnostics;

namespace VeriStandController
{
    internal class Program
    {
        public static void Main(string[] args)
        {
            // Create an array of channel names, as defined in the system definition file
            string[] names = {
                "Aliases/ActualRPM",
                "Aliases/DesiredRPM"
            };

            // An array whose elements are the values of each channel
            double[] values = new double[names.Length];

            string gatewayIp = "localhost";
            string systemDefinitionPath = @"C:\Users\Public\Documents\National Instruments\NI VeriStand 2024\Examples\Stimulus Profile\Engine Demo\Engine Demo.nivssdf";

            Factory factory = new Factory();
            IWorkspace2 workspace = factory.GetIWorkspace2(gatewayIp);

            // Connect to the system and deploy the system definition file
            workspace.ConnectToSystem(systemDefinitionPath, true, 60000); // ms

            int totalMessages = 100000;
            int messageCount = 0;
            int messagesLast5Seconds = 0;

            Stopwatch totalStopwatch = new Stopwatch();
            Stopwatch intervalStopwatch = new Stopwatch();

            totalStopwatch.Start();
            intervalStopwatch.Start();

            while (messageCount < totalMessages)
            {
                workspace.GetMultipleChannelValues(names, out values);
                messageCount++;
                messagesLast5Seconds++;

                // Check if 5 seconds have passed
                if (intervalStopwatch.ElapsedMilliseconds >= 5000)
                {
                    double messagesPerSecond = messagesLast5Seconds / (intervalStopwatch.ElapsedMilliseconds / 1000.0);
                    Console.WriteLine($"Messages per second (last 5 seconds): {messagesPerSecond:F2}");

                    // Reset the 5-second counter
                    messagesLast5Seconds = 0;
                    intervalStopwatch.Restart();
                }
            }

            totalStopwatch.Stop();
            double totalTime = totalStopwatch.ElapsedMilliseconds / 1000.0;
            double averageMessagesPerSecond = totalMessages / totalTime;

            Console.WriteLine($"\nTotal time: {totalTime:F2} seconds");
            Console.WriteLine($"Average messages per second: {averageMessagesPerSecond:F2}");
        }
    }
}
