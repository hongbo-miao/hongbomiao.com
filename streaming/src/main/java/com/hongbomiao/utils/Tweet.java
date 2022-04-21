package com.hongbomiao.utils;

import java.sql.Timestamp;

public final class Tweet {
  public Tweet(Timestamp timestamp, String id, String idStr, String text, String lang) {
    this.timestamp = timestamp;
    this.id = id;
    this.idStr = idStr;
    this.text = text;
    this.lang = lang;
  }

  public final Timestamp timestamp;
  public final String id;
  public final String idStr;
  public final String text;
  public final String lang;
}
