from nptdms import TdmsFile


def main():
    tdms_file = TdmsFile.read("data/exampleMeasurements.tdms")
    for group in tdms_file.groups():
        df = group.as_dataframe()
        print(group.name)
        print(df)


if __name__ == "__main__":
    main()
