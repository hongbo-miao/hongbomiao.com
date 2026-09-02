package com.hongbomiao.utils;

import io.github.cdimascio.dotenv.Dotenv;
import io.github.cdimascio.dotenv.DotenvEntry;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Config {
  public Config() {
    Map<String, String> environmentValues = readEnvironmentFiles();
    this.swimHost = getRequired(environmentValues, "SWIM_HOST");
    this.swimVPNName = getRequired(environmentValues, "SWIM_VPN_NAME");
    this.swimUsername = getRequired(environmentValues, "SWIM_USERNAME");
    this.swimPassword = getRequired(environmentValues, "SWIM_PASSWORD");
    this.swimQueueName = getRequired(environmentValues, "SWIM_QUEUE_NAME");
  }

  public final String swimHost;
  public final String swimVPNName;
  public final String swimUsername;
  public final String swimPassword;
  public final String swimQueueName;

  private static List<String> getEnvironmentFileNames() {
    String environment = System.getenv("ENVIRONMENT");
    return switch (environment == null ? "" : environment) {
      case "development", "test" -> List.of(".env.development", ".env.development.local");
      case "production" -> List.of(".env.production", ".env.production.local");
      default -> throw new IllegalStateException("Invalid ENVIRONMENT value: " + environment + ".");
    };
  }

  private static Map<String, String> readEnvironmentFiles() {
    Map<String, String> environmentValues = new LinkedHashMap<>();
    for (String environmentFileName : getEnvironmentFileNames()) {
      Dotenv dotenv = Dotenv.configure()
          .filename(environmentFileName)
          .ignoreIfMissing()
          .load();
      for (DotenvEntry entry : dotenv.entries(Dotenv.Filter.DECLARED_IN_ENV_FILE)) {
        environmentValues.put(entry.getKey(), entry.getValue());
      }
    }
    return environmentValues;
  }

  private static String getRequired(Map<String, String> environmentValues, String name) {
    String value = System.getenv(name);
    if (value == null || value.isBlank()) {
      value = environmentValues.get(name);
    }
    if (value == null || value.isBlank()) {
      throw new IllegalStateException(name + " is not set");
    }
    return value;
  }
}
