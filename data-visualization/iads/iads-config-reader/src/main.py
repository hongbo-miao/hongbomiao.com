import logging

import pythoncom
import win32com.client


def show_version_from_file(iads_config, filename):
    try:
        version = iads_config.VersionFromFile(filename)
        logging.info(f"Version from file: {version}")
    except Exception as e:
        logging.error(f"Could not get version from {filename}: {e}")


def execute_query(iads_config, query):
    try:
        logging.info(f"Executing: {query}")
        results = iads_config.Query(query)
        if results:
            for result in results:
                logging.info(f"Result: {result}")
    except Exception as e:
        logging.error(f"Query failed: {e}")


def main():
    iads_config = None
    try:
        pythoncom.CoInitialize()
        iads_config = win32com.client.Dispatch("IadsConfigInterface.IadsConfig")

        show_version_from_file(iads_config, "pfConfig")

        iads_config.Open("pfConfig", True)

        execute_query(iads_config, "select * from Desktops")
        execute_query(iads_config, "select System.RowNumber from Desktops")

        iads_config.Close(True)

    except Exception as e:
        logging.error(f"Error occurred: {e}")
    finally:
        # Clean up COM resources
        if iads_config:
            iads_config = None
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
