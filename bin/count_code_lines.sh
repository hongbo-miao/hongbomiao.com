#!/usr/bin/env bash
set -e

# Get the location of this script
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )/.."

# Run cloc - this counts code lines, blank lines and comment lines for the specified languages.
# We are only interested in the summary, therefore the tail -1
SUMMARY="$(cloc "${SCRIPT_DIR}" --md | tail -1)"

# The $SUMMARY is one line of a markdown table which looks like: SUM:|101|3123|2238|10783
# We use the following command to split it into an array
IFS='|' read -r -a TOKENS <<< "$SUMMARY"

# Store the individual tokens for better readability
CODE_LINE_COUNT="${TOKENS[4]}"

awk -v a="$CODE_LINE_COUNT" 'BEGIN {printf "%.1fk\n", a/1000}'
