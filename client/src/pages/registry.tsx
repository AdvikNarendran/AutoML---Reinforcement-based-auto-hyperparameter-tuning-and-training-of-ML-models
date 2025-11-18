import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Download, Cpu, TrendingUp, Calendar } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Model } from "@shared/schema";

export default function Registry() {
  const { data: models } = useQuery<Model[]>({
    queryKey: ["/api/models"],
  });

  const sortedModels = models?.sort((a, b) =>
    new Date(b.trainedAt).getTime() - new Date(a.trainedAt).getTime()
  );

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle className="text-xl font-semibold">Model Registry</CardTitle>
          <p className="text-sm text-muted-foreground">
            All trained models with their performance metrics and metadata
          </p>
        </CardHeader>
        <CardContent>
          {sortedModels && sortedModels.length > 0 ? (
            <div className="rounded-md border border-border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow className="bg-muted/50">
                    <TableHead className="font-semibold">Model ID</TableHead>
                    <TableHead className="font-semibold">Name</TableHead>
                    <TableHead className="font-semibold">Type</TableHead>
                    <TableHead className="font-semibold">Algorithm</TableHead>
                    <TableHead className="font-semibold text-right">Accuracy</TableHead>
                    <TableHead className="font-semibold text-right">F1 Score</TableHead>
                    <TableHead className="font-semibold">Status</TableHead>
                    <TableHead className="font-semibold">Trained At</TableHead>
                    <TableHead className="font-semibold">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedModels.map((model) => (
                    <TableRow key={model.id} className="hover-elevate" data-testid={`model-row-${model.id}`}>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {model.id.slice(0, 8)}...
                      </TableCell>
                      <TableCell className="font-medium">{model.name}</TableCell>
                      <TableCell>
                        <Badge variant="secondary">{model.type}</Badge>
                      </TableCell>
                      <TableCell className="font-mono text-sm">{model.algorithm}</TableCell>
                      <TableCell className="text-right font-semibold">
                        {(model.metrics.accuracy * 100).toFixed(2)}%
                      </TableCell>
                      <TableCell className="text-right font-semibold">
                        {(model.metrics.f1Score * 100).toFixed(2)}%
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            model.status === "completed"
                              ? "default"
                              : model.status === "training"
                              ? "secondary"
                              : "destructive"
                          }
                        >
                          {model.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {new Date(model.trainedAt).toLocaleString()}
                      </TableCell>
                      <TableCell>
                        <Button
                          size="sm"
                          variant="ghost"
                          data-testid={`button-view-${model.id}`}
                        >
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="text-center py-16 text-muted-foreground">
              <Cpu className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <div className="text-sm font-medium">No models in registry</div>
              <div className="text-xs mt-1">Train a model to see it here</div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Model Statistics */}
      {sortedModels && sortedModels.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Models
              </CardTitle>
              <Cpu className="w-4 h-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-foreground">{sortedModels.length}</div>
              <p className="text-xs text-muted-foreground mt-1">
                {sortedModels.filter(m => m.status === "completed").length} completed
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Best Accuracy
              </CardTitle>
              <TrendingUp className="w-4 h-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-foreground">
                {(Math.max(...sortedModels.map(m => m.metrics.accuracy)) * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Across all models
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Latest Training
              </CardTitle>
              <Calendar className="w-4 h-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-foreground">
                {new Date(sortedModels[0].trainedAt).toLocaleDateString()}
              </div>
              <p className="text-xs text-muted-foreground mt-1 font-mono">
                {new Date(sortedModels[0].trainedAt).toLocaleTimeString()}
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
