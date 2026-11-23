import { getLeaderboard, LeaderboardEntry } from '@/lib/api';
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, FlatList, RefreshControl, SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

type Window = 'all' | '30d' | '365d';
const WINDOWS: { key: Window; label: string }[] = [
  { key: '30d', label: '30 days' },
  { key: '365d', label: '1 year' },
  { key: 'all', label: 'All time' },
];

export default function Leaderboard() {
  const [window, setWindow] = useState<Window>('30d');
  const [data, setData] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await getLeaderboard(window));
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load leaderboard');
    } finally {
      setLoading(false);
    }
  }, [window]);

  useEffect(() => { load(); }, [load]);

  const renderItem = ({ item }: { item: LeaderboardEntry }) => {
    // Top 3 get special metallic colors
    let rankColor = '#888'; // Default rank color
    if (item.rank === 1) rankColor = '#FFD700'; // Gold
    if (item.rank === 2) rankColor = '#C0C0C0'; // Silver
    if (item.rank === 3) rankColor = '#CD7F32'; // Bronze

    return (
      <View style={styles.card}>
        <View style={styles.rankContainer}>
          <Text style={[styles.rankText, { color: rankColor }]}>
            #{item.rank}
          </Text>
        </View>

        <View style={styles.avatarPlaceholder}>
          <Text style={styles.avatarInitials}>{item.name.charAt(0).toUpperCase()}</Text>
        </View>

        <View style={styles.userInfo}>
          <Text style={styles.nameText} numberOfLines={1}>
            {item.name}
          </Text>
          <Text style={styles.subText}>{item.resolved_count} resolved</Text>
        </View>

        <View style={styles.statContainer}>
          <View style={styles.statBadge}>
            <Text style={styles.statNumber}>{item.report_count}</Text>
            <Text style={styles.statLabel}>Reports</Text>
          </View>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <View style={styles.iconContainer}>
              { /* <IconSymbol size={24} name="sparkles" color="#2DD4BF" /> */}
            </View>
            <Text style={styles.headerTitle}>Leaderboard</Text>
          </View>
          <View style={styles.activeBadge}>
            <Text style={styles.activeBadgeText}>Top Reporters</Text>
          </View>
        </View>

        <View style={styles.windowRow}>
          {WINDOWS.map(w => (
            <TouchableOpacity
              key={w.key}
              onPress={() => setWindow(w.key)}
              style={[styles.windowBtn, window === w.key && styles.windowBtnActive]}
            >
              <Text style={[styles.windowText, window === w.key && styles.windowTextActive]}>{w.label}</Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.listWrapper}>
          {loading && data.length === 0 ? (
            <ActivityIndicator style={{ marginTop: 40 }} color="#2DD4BF" />
          ) : error ? (
            <Text style={styles.emptyText}>{error}</Text>
          ) : (
            <FlatList
              data={data}
              keyExtractor={(item) => item.id}
              renderItem={renderItem}
              contentContainerStyle={styles.listContainer}
              showsVerticalScrollIndicator={false}
              refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor="#2DD4BF" />}
              ListEmptyComponent={<Text style={styles.emptyText}>No reports yet. Be the first!</Text>}
            />
          )}
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  windowRow: { flexDirection: 'row', gap: 8, paddingHorizontal: 20, paddingTop: 16 },
  windowBtn: {
    paddingHorizontal: 14, paddingVertical: 6, borderRadius: 16,
    borderWidth: 1, borderColor: '#27272a',
  },
  windowBtnActive: { backgroundColor: 'rgba(20, 184, 166, 0.15)', borderColor: '#2DD4BF' },
  windowText: { color: '#a1a1aa', fontSize: 13, fontWeight: '600' },
  windowTextActive: { color: '#2DD4BF' },
  subText: { color: '#71717a', fontSize: 12, marginTop: 2 },
  emptyText: { color: '#a1a1aa', textAlign: 'center', marginTop: 40 },
  safeArea: {
    flex: 1,
    backgroundColor: '#09090b', // Base dark background
  },
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 40, // Extra padding for status bar if not handled by SafeAreaView perfectly
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#27272a',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 8,
    backgroundColor: '#111827', // Darker offset background for the icon
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
    borderWidth: 1,
    borderColor: '#1f2937',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  activeBadge: {
    backgroundColor: 'rgba(20, 184, 166, 0.1)', // #14b8a6 with 10% opacity
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(20, 184, 166, 0.2)',
  },
  activeBadgeText: {
    color: '#2DD4BF', // Teal accent
    fontSize: 12,
    fontWeight: '600',
  },
  listWrapper: {
    flex: 1,
    padding: 16,
  },
  listContainer: {
    paddingBottom: 20,
  },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0f0f11', // Very dark gray, slightly lighter than background
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#27272a', // Subtle dashed border look can be achieved with a solid subtle border here, as dashed per-side is tricky in RN without SVG
    borderStyle: 'dashed', // Applying dashed border explicitly as requested
  },
  rankContainer: {
    width: 40,
    alignItems: 'center',
    marginRight: 8,
  },
  rankText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    marginRight: 16,
  },
  avatarPlaceholder: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#27272a',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  avatarInitials: {
    color: '#a1a1aa',
    fontSize: 20,
    fontWeight: '600',
  },
  userInfo: {
    flex: 1,
  },
  nameText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 4,
  },
  statContainer: {
    marginLeft: 12,
  },
  statBadge: {
    alignItems: 'center',
    backgroundColor: 'rgba(20, 184, 166, 0.1)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    minWidth: 70,
  },
  statNumber: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2DD4BF',
  },
  statLabel: {
    fontSize: 10,
    color: '#a1a1aa',
    marginTop: 2,
    textTransform: 'uppercase',
  },
});
