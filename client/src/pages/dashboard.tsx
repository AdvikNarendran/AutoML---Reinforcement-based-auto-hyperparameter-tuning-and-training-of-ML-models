import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Database, Cpu, RefreshCw, TrendingUp, Activity, Award } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Badge } from "@/components/ui/badge";
import type { Model, Episode, Log } from "@shared/schema";

export default function Dashboard() {
  const { data: models } = useQuery<Model[]>({
    queryKey: ["/api/models"],
  });

  const { data: episodes } = useQuery<Episode[]>({
    queryKey: ["/api/episodes"],
  });

  const { data: logs } = useQuery<Log[]>({
    queryKey: ["/api/logs"],
  });

  // Calculate KPIs
  const currentModel = models?.find(m => m.status === "completed");
  const currentAccuracy = currentModel?.metrics.accuracy || 0;
  const totalModels = models?.length || 0;
  const completedEpisodes = episodes?.filter(e => e.status === "completed") || [];
  const avgReward = completedEpisodes.reduce((sum, e) => sum + e.totalReward, 0) / (completedEpisodes.length || 1);
  const retrainEvents = completedEpisodes.reduce((sum, e) => sum + e.retrainCount, 0);

  // Performance chart data
  const performanceData = completedEpisodes.slice(-10).map((ep, idx) => ({
    episode: ep.episodeNumber,
    accuracy: ep.finalAccuracy * 100,
    reward: ep.totalReward,
  }));

  return (
    <div className="space-y-8">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Current Accuracy
            </CardTitle>
            <Activity className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-foreground" data-testid="text-current-accuracy">
              {(currentAccuracy * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {currentModel ? "Active model" : "No trained model"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Models Trained
            </CardTitle>
            <Cpu className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-foreground" data-testid="text-total-models">
              {totalModels}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Total in registry
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Retraining Events
            </CardTitle>
            <RefreshCw className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-foreground" data-testid="text-retrain-events">
              {retrainEvents}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Across {completedEpisodes.length} episodes
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Avg Reward
            </CardTitle>
            <Award className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-foreground" data-testid="text-avg-reward">
              {avgReward.toFixed(2)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Per episode
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Performance Chart & Recent Episodes */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Performance Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            {performanceData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                  <XAxis
                    dataKey="episode"
                    className="text-xs text-muted-foreground"
                    label={{ value: "Episode", position: "insideBottom", offset: -5 }}
                  />
                  <YAxis
                    className="text-xs text-muted-foreground"
                    label={{ value: "Accuracy (%)", angle: -90, position: "insideLeft" }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "var(--radius)",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-muted-foreground text-sm">
                No episode data available yet
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Recent Episodes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {completedEpisodes.slice(-5).reverse().map((episode) => (
                <div
                  key={episode.id}
                  className="flex items-center justify-between gap-4 p-3 rounded-md bg-accent/50"
                  data-testid={`episode-${episode.id}`}
                >
                  <div className="flex-1">
                    <div className="text-sm font-medium font-mono">Episode {episode.episodeNumber}</div>
                    <div className="text-xs text-muted-foreground">
                      {episode.totalSteps} steps • {episode.retrainCount} retrains
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold">{(episode.finalAccuracy * 100).toFixed(1)}%</div>
                    <div className="text-xs text-muted-foreground">+{episode.totalReward.toFixed(2)} reward</div>
                  </div>
                </div>
              ))}
              {completedEpisodes.length === 0 && (
                <div className="text-sm text-muted-foreground text-center py-8">
                  No episodes completed yet
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Logs Preview */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2">
          <CardTitle className="text-lg font-semibold">Recent Activity</CardTitle>
          <a href="/logs" className="text-xs text-primary hover:underline" data-testid="link-view-all-logs">
            View All →
          </a>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {logs?.slice(-5).reverse().map((log) => (
              <div
                key={log.id}
                className="flex items-start gap-3 p-2 rounded-md hover-elevate"
                data-testid={`log-${log.id}`}
              >
                <Badge
                  variant={log.level === "error" ? "destructive" : log.level === "warning" ? "secondary" : "default"}
                  className="mt-0.5"
                >
                  {log.level}
                </Badge>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-foreground">{log.component}</div>
                  <div className="text-xs text-muted-foreground truncate">{log.message}</div>
                </div>
                <div className="text-xs text-muted-foreground font-mono whitespace-nowrap">
                  {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : ""}
                </div>
              </div>
            ))}
            {(!logs || logs.length === 0) && (
              <div className="text-sm text-muted-foreground text-center py-8">
                No activity logged yet
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
