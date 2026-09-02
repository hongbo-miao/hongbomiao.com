package com.hongbomiao;

import com.hongbomiao.utils.Config;
import com.solacesystems.jcsmp.BytesXMLMessage;
import com.solacesystems.jcsmp.ConsumerFlowProperties;
import com.solacesystems.jcsmp.FlowReceiver;
import com.solacesystems.jcsmp.JCSMPChannelProperties;
import com.solacesystems.jcsmp.JCSMPException;
import com.solacesystems.jcsmp.JCSMPFactory;
import com.solacesystems.jcsmp.JCSMPProperties;
import com.solacesystems.jcsmp.JCSMPSession;
import com.solacesystems.jcsmp.Queue;
import com.solacesystems.jcsmp.SDTException;
import com.solacesystems.jcsmp.TextMessage;
import com.solacesystems.jcsmp.XMLMessageListener;
import java.nio.charset.StandardCharsets;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

public class Main {
  public static void main(String[] args) throws Exception {
    Config config = new Config();

    JCSMPSession session = createSwimSession(config);
    session.connect();
    System.out.println("Connected to SWIM Cloud Distribution Service.");

    Queue queue = JCSMPFactory.onlyInstance().createQueue(config.swimQueueName);
    ConsumerFlowProperties flowProperties = new ConsumerFlowProperties();
    flowProperties.setEndpoint(queue);
    flowProperties.setAckMode(JCSMPProperties.SUPPORTED_MESSAGE_ACK_CLIENT);

    AtomicBoolean isShuttingDown = new AtomicBoolean(false);

    // Messages arrive on a JCSMP dispatch thread rather than through a blocking receive(), because receive() holds an internal queue lock while it waits, which deadlocks any attempt to close the session from another thread such as a shutdown hook.
    // Null endpoint properties means we never try to provision the queue, since SCDS owns it and the client has no provision rights.
    FlowReceiver flowReceiver = session.createFlow(new SwimMessageListener(isShuttingDown), flowProperties, null);
    flowReceiver.start();
    System.out.println("Waiting for messages...");

    CountDownLatch shutdownLatch = new CountDownLatch(1);
    Runtime.getRuntime().addShutdownHook(new Thread(() -> {
      isShuttingDown.set(true);
      flowReceiver.close();
      session.closeSession();
      shutdownLatch.countDown();
    }));
    shutdownLatch.await();
  }

  private static JCSMPSession createSwimSession(Config config) throws Exception {
    JCSMPProperties properties = new JCSMPProperties();
    properties.setProperty(JCSMPProperties.HOST, config.swimHost);
    properties.setProperty(JCSMPProperties.VPN_NAME, config.swimVPNName);
    properties.setProperty(JCSMPProperties.USERNAME, config.swimUsername);
    properties.setProperty(JCSMPProperties.PASSWORD, config.swimPassword);

    JCSMPChannelProperties channelProperties = (JCSMPChannelProperties)
        properties.getProperty(JCSMPProperties.CLIENT_CHANNEL_PROPERTIES);
    channelProperties.setConnectRetries(1);
    channelProperties.setReconnectRetries(-1);

    return JCSMPFactory.onlyInstance().createSession(properties);
  }

  private static final class SwimMessageListener implements XMLMessageListener {
    private final AtomicBoolean isShuttingDown;

    SwimMessageListener(AtomicBoolean isShuttingDown) {
      this.isShuttingDown = isShuttingDown;
    }

    @Override
    public void onReceive(BytesXMLMessage message) {
      printMessage(message);
      // Client ack, sent only after the message is printed, so nothing is lost on a crash.
      message.ackMessage();
    }

    @Override
    public void onException(JCSMPException exception) {
      // A flow error while shutting down is expected (closing the flow triggers one) and not a failure.
      if (isShuttingDown.get()) {
        return;
      }
      // A lost or broken flow otherwise means messages can silently stop arriving, so
      // fail loudly and exit rather than run on in a state that looks alive but is not.
      System.err.println("Flow error: " + exception.getMessage());
      System.exit(1);
    }
  }

  private static void printMessage(BytesXMLMessage message) {
    System.out.println("----------------------------------------");
    System.out.println("Application message id: " + message.getApplicationMessageId());
    System.out.println("Sender timestamp: " + message.getSenderTimestamp());

    if (message.getProperties() != null) {
      Set<String> propertyNames = message.getProperties().keySet();
      for (String propertyName : propertyNames) {
        try {
          System.out.println("Property " + propertyName + ": " + message.getProperties().get(propertyName));
        } catch (SDTException exception) {
          System.out.println("Property " + propertyName + ": <unreadable> " + exception.getMessage());
        }
      }
    }

    if (message instanceof TextMessage textMessage) {
      System.out.println(textMessage.getText());
    } else {
      System.out.println(new String(message.getBytes(), StandardCharsets.UTF_8));
    }
  }
}
