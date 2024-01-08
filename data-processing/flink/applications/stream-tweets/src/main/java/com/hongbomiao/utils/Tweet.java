package com.hongbomiao.utils;

import java.sql.Timestamp;

public final class Tweet {
  public Tweet(Timestamp timestamp, String id, String twitterUserID, String text, String lang) {
    this.timestamp = timestamp;
    this.id = id;
    this.twitterUserID = twitterUserID;
    this.text = text;
    this.lang = lang;
  }

  public final Timestamp timestamp;
  public final String id;
  public final String twitterUserID;
  public final String text;
  public final String lang;
}
