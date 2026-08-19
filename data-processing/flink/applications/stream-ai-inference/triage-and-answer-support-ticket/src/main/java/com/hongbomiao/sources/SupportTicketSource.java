package com.hongbomiao.sources;

import java.util.Random;
import org.apache.flink.connector.datagen.source.GeneratorFunction;
import org.apache.flink.types.Row;

/** Generates a synthetic customer-support ticket, tagged with a customer and a service component. */
public class SupportTicketSource implements GeneratorFunction<Long, Row> {

  private record TicketTemplate(String component, String ticketText) {}

  // Components line up with the topics in knowledge-base.json, so retrieval stays meaningful, and
  // with the components ServiceIncidentSource cycles through, so a ticket can land mid-outage.
  private static final TicketTemplate[] TICKET_TEMPLATES = {
    new TicketTemplate("billing", "My card was declined when I tried to renew my annual plan, what happened?"),
    new TicketTemplate("authentication", "I forgot my password and the reset email never arrived, please help urgently."),
    new TicketTemplate("mobile", "The app crashes every time it opens since the latest update, this is urgent, I cannot work."),
    new TicketTemplate("account", "How do I add a new teammate to our workspace?"),
    new TicketTemplate("billing", "My invoice this month has the wrong tax amount, can you fix it?"),
    new TicketTemplate("authentication", "Two-factor authentication codes are not arriving on my phone."),
    new TicketTemplate("billing", "I would like to cancel my subscription at the end of this cycle."),
    new TicketTemplate("webhooks", "Our webhook deliveries have been failing all night, this is urgent for our production system."),
    new TicketTemplate("billing", "Can I get a refund for the annual plan I just purchased yesterday?"),
    new TicketTemplate("api", "The API keeps returning 429 errors when we call it in a loop."),
    new TicketTemplate("authentication", "How do I enable single sign-on for our enterprise workspace?"),
    new TicketTemplate("billing", "My account was suspended after a payment failure, how do I reactivate it?")
  };

  // Small enough that the same customer's tickets recur within minutes, so customer_memory
  // visibly accumulates.
  private static final String[] CUSTOMER_IDS = {
    "customer-1", "customer-2", "customer-3", "customer-4", "customer-5"
  };

  private transient Random random;

  @Override
  public void open(org.apache.flink.api.connector.source.SourceReaderContext readerContext) {
    random = new Random();
  }

  @Override
  public Row map(Long sequenceNumber) {
    if (random == null) {
      random = new Random();
    }
    TicketTemplate template = TICKET_TEMPLATES[random.nextInt(TICKET_TEMPLATES.length)];
    String customerId = CUSTOMER_IDS[random.nextInt(CUSTOMER_IDS.length)];
    return Row.of(sequenceNumber, customerId, template.component(), template.ticketText());
  }
}
