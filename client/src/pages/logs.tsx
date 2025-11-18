import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, FileText, AlertCircle, Info, AlertTriangle } from "lucide-react";
import type { Log } from "@shared/schema";

export default function Logs() {
  const [searchTerm, setSearchTerm] = useState("");
  const [levelFilter, setLevelFilter] = useState<string>("all");
  const [componentFilter, setComponentFilter] = useState<string>("all");

  const { data: logs } = useQuery<Log[]>({
    queryKey: ["/api/logs"],
  });

  // Filter logs
  const filteredLogs = logs?.filter(log => {
    const matchesSearch = searchTerm === "" ||
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.component.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesLevel = levelFilter === "all" || log.level === levelFilter;
    const matchesComponent = componentFilter === "all" || log.component === componentFilter;
    return matchesSearch && matchesLevel && matchesComponent;
  });

  // Get unique components
  const components = Array.from(new Set(logs?.map(l => l.component) || []));

  const getLevelIcon = (level: string) => {
    switch (level) {
      case "error":
        return <AlertCircle className="w-4 h-4" />;
      case "warning":
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <Info className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Log Filters</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search logs..."
                className="pl-9"
                data-testid="input-search-logs"
              />
            </div>

            <Select value={levelFilter} onValueChange={setLevelFilter}>
              <SelectTrigger data-testid="select-log-level">
                <SelectValue placeholder="Filter by level" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Levels</SelectItem>
                <SelectItem value="info">Info</SelectItem>
                <SelectItem value="warning">Warning</SelectItem>
                <SelectItem value="error">Error</SelectItem>
              </SelectContent>
            </Select>

            <Select value={componentFilter} onValueChange={setComponentFilter}>
              <SelectTrigger data-testid="select-log-component">
                <SelectValue placeholder="Filter by component" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Components</SelectItem>
                {components.map(comp => (
                  <SelectItem key={comp} value={comp}>{comp}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Showing {filteredLogs?.length || 0} of {logs?.length || 0} logs</span>
          </div>
        </CardContent>
      </Card>

      {/* Logs Display */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Activity Logs</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {filteredLogs && filteredLogs.length > 0 ? (
              filteredLogs.slice().reverse().map((log) => (
                <div
                  key={log.id}
                  className="p-4 rounded-md border border-border hover-elevate"
                  data-testid={`log-entry-${log.id}`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`mt-1 ${
                      log.level === "error" ? "text-destructive" :
                      log.level === "warning" ? "text-yellow-600" :
                      "text-muted-foreground"
                    }`}>
                      {getLevelIcon(log.level)}
                    </div>

                    <div className="flex-1 min-w-0 space-y-2">
                      <div className="flex items-center gap-2 flex-wrap">
                        <Badge
                          variant={
                            log.level === "error" ? "destructive" :
                            log.level === "warning" ? "secondary" :
                            "default"
                          }
                          className="text-xs"
                        >
                          {log.level.toUpperCase()}
                        </Badge>
                        <Badge variant="outline" className="text-xs font-mono">
                          {log.component}
                        </Badge>
                        <span className="text-xs text-muted-foreground font-mono ml-auto">
                          {log.timestamp ? new Date(log.timestamp).toLocaleString() : ""}
                        </span>
                      </div>

                      <div className="text-sm text-foreground font-medium">
                        {log.message}
                      </div>

                      {log.metadata && Object.keys(log.metadata).length > 0 && (
                        <details className="text-xs">
                          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                            View metadata
                          </summary>
                          <pre className="mt-2 p-3 rounded-md bg-muted/50 font-mono text-xs overflow-x-auto">
                            {JSON.stringify(log.metadata, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-16 text-muted-foreground">
                <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <div className="text-sm font-medium">
                  {logs && logs.length > 0 ? "No logs match your filters" : "No logs available"}
                </div>
                <div className="text-xs mt-1">
                  {logs && logs.length > 0 ? "Try adjusting your filter criteria" : "Activity will appear here"}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
