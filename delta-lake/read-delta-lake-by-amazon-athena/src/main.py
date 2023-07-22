import awswrangler as wr


def main():
    df = wr.athena.read_sql_query(
        "select * from motor limit 10;",
        database="hm_iot_db",
    )
    print(df)


if __name__ == "__main__":
    main()
