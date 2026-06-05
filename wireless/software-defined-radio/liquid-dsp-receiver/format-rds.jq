# Collapse redsea's per-group JSON stream into concise, change-only lines:
#   Station: <call sign or PI>   stable identity (NOT the scrolling PS field)
#   Song:    <artist - title>    from RadioText+ tags, else the raw RadioText
#
# Two stations habits make a naive filter noisy, so this avoids both:
#   - The 8-char PS field is often scrolled ("WiLD 949", "Alex", "Warren"...),
#     so the station name is taken from the constant PI / call sign instead.
#   - Now-playing is sent twice (structured RadioText+ and free-text RadioText);
#     once any RadioText+ is seen, the free-text RadioText is ignored so the
#     Song line stops flipping between the two formats.
#
# foreach carries the last station/song forward so a line prints only on change.
foreach inputs as $group (
  {station: null, song: null, seen_rtplus: false};
  (.seen_rtplus or ($group.radiotext_plus.tags != null)) as $seen
  | ($group.callsign // $group.pi // .station) as $station
  | (
      if $group.radiotext_plus.tags then
        [$group.radiotext_plus.tags[] | .data] | join(" - ")
      elif (($seen | not) and ($group.radiotext != null)) then
        $group.radiotext
      else
        .song
      end
    ) as $song
  | {
      station: $station,
      song: $song,
      seen_rtplus: $seen,
      lines: [
        (if $station != null and $station != .station then "Station: \($station)" else empty end),
        (if $song != null and $song != .song then "Song:    \($song)" else empty end)
      ]
    };
  .lines[]
)
