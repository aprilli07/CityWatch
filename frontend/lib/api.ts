import * as Location from 'expo-location';
import { Platform } from 'react-native';

// The API is the only backend the app talks to: photo upload, classification,
// weather and storage all happen server-side, so no database keys ship in the app.
// - iOS simulator: http://localhost:8000
// - Android emulator: http://10.0.2.2:8000
// - Physical device: http://<your machine's LAN IP>:8000
export const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_URL ??
  (Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API ${res.status} on ${path}: ${text.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

// ─── Reports ────────────────────────────────────────────────────────────────

export type ReportStatus = 'open' | 'in_progress' | 'resolved' | 'rejected';

export type Report = {
  id: string;
  image_url: string | null;
  issue_type: string | null;
  cv_confidence: number | null;
  severity: number | null;
  status: ReportStatus | null;
  created_at: string;
  latitude: number | null;
  longitude: number | null;
  h3_cell: string | null;
};

export type SubmitReportResult = {
  success: boolean;
  report: Report;
  incident_id: number | null;
  classification: { prediction: string | null; confidence: number | null; top3: { label: string; p: number }[] };
  severity_source: 'llm' | 'heuristic';
};

export async function submitPotholeReport(
  photoUri: string,
  location: Location.LocationObject,
  photoTimestamp: number, // ms since epoch, from EXIF DateTime or Date.now()
  userId?: string,
): Promise<SubmitReportResult> {
  const form = new FormData();
  // React Native's FormData accepts { uri, name, type } file descriptors.
  form.append('file', { uri: photoUri, name: 'report.jpg', type: 'image/jpeg' } as any);
  form.append('latitude', String(location.coords.latitude));
  form.append('longitude', String(location.coords.longitude));
  form.append('photo_timestamp', new Date(photoTimestamp).toISOString());
  if (userId) form.append('user_id', userId);
  return request<SubmitReportResult>('/api/reports', { method: 'POST', body: form });
}

export async function getMyReports(userId: string): Promise<Report[]> {
  const json = await request<{ data: Report[] }>(`/api/reports?user_id=${encodeURIComponent(userId)}`);
  return json.data;
}

// ─── Users ──────────────────────────────────────────────────────────────────

export async function saveUser(id: string, email?: string, displayName?: string) {
  return request('/api/users', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, email, display_name: displayName }),
  });
}

// ─── Stats ──────────────────────────────────────────────────────────────────

export type StatsData = {
  active_count: number;
  resolved_this_week: number;
  reported_this_week: number;
  total_count: number;
  median_days_to_resolve: number | null;
  issue_type_counts: Record<string, number>;
};

export async function getStats(): Promise<StatsData> {
  return (await request<{ data: StatsData }>('/api/stats')).data;
}

// ─── Heatmap / regions ──────────────────────────────────────────────────────

export type HeatmapPeriod = 'current' | '30d' | '90d' | '365d';

export type HeatmapCell = {
  h3_cell: string;
  latitude: number;
  longitude: number;
  asset_count: number;
  issue_count: number;   // open reports in the region
  value: number;         // 0-1 intensity for the selected period
  risk: number;          // raw value (severity, or failure probability)
  avg_severity: number;
  predict_30d: number;
  predict_90d: number;
  predict_365d: number;
};

export async function getHeatmap(period: HeatmapPeriod = 'current'): Promise<HeatmapCell[]> {
  return (await request<{ data: HeatmapCell[] }>(`/api/heatmap?period=${period}`)).data;
}

export async function getRegionsGeoJSON(
  horizon: Exclude<HeatmapPeriod, 'current'>,
): Promise<GeoJSON.FeatureCollection> {
  return request<GeoJSON.FeatureCollection>(`/api/regions?horizon=${horizon}`);
}

// ─── Leaderboard ────────────────────────────────────────────────────────────

export type LeaderboardEntry = {
  id: string;
  name: string;
  report_count: number;
  resolved_count: number;
  rank: number;
};

export async function getLeaderboard(window: 'all' | '30d' | '365d' = 'all', limit = 20) {
  return (await request<{ data: LeaderboardEntry[] }>(`/api/leaderboard?window=${window}&limit=${limit}`)).data;
}
