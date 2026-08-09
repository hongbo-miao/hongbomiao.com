export default async function fetchHlsPlaylistUrl(): Promise<string> {
  const response = await fetch('/stream');
  if (!response.ok) {
    throw new Error(`Failed to fetch HLS playlist URL: ${response.statusText}`);
  }
  const data = (await response.json()) as { playlist_url: string };
  return data.playlist_url;
}
