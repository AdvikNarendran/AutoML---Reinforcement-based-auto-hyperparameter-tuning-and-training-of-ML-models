import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Play, Zap, TrendingUp, Target } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { Dataset, Model, OptunaTrial } from "@shared/schema";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from "recharts";

export default function Training() {
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [algorithm, setAlgorithm] = useState("LogisticRegression");
  const [optunaTrials, setOptunaTrials] = useState(20);
  const { toast } = useToast();

  const { data: datasets } = useQuery<Dataset[]>({
    queryKey: ["/api/datasets"],
  });

  const { data: models } = useQuery<Model[]>({
    queryKey: ["/api/models"],
  });

  const { data: trials } = useQuery<OptunaTrial[]>({
    queryKey: ["/api/optuna/trials"],
  });

  const baselineTrainMutation = useMutation({
    mutationFn: async (data: { datasetId: string; algorithm: string }) =>
      apiRequest("POST", "/api/training/baseline", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/models"] });
      queryClient.invalidateQueries({ queryKey: ["/api/logs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/episodes"] });
      toast({
        title: "Baseline training completed",
        description: "Model has been trained successfully",
      });
    },
    onError: () => {
      toast({
        title: "Training failed",
        description: "There was an error starting the training",
        variant: "destructive",
      });
    },
  });

  const optunaTuneMutation = useMutation({
    mutationFn: async (data: { datasetId: string; algorithm: string; nTrials: number }) =>
      apiRequest("POST", "/api/training/optuna", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/models"] });
      queryClient.invalidateQueries({ queryKey: ["/api/optuna/trials"] });
      queryClient.invalidateQueries({ queryKey: ["/api/logs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/episodes"] });
      toast({
        title: "Hyperparameter tuning completed",
        description: "Optuna has found the best parameters",
      });
    },
    onError: () => {
      toast({
        title: "Tuning failed",
        description: "There was an error starting hyperparameter tuning",
        variant: "destructive",
      });
    },
  });

  const handleBaselineTrain = () => {
    if (!selectedDataset) {
      toast({
        title: "No dataset selected",
        description: "Please select a dataset first",
        variant: "destructive",
      });
      return;
    }
    baselineTrainMutation.mutate({ datasetId: selectedDataset, algorithm });
  };

  const handleOptunaTune = () => {
    if (!selectedDataset) {
      toast({
        title: "No dataset selected",
        description: "Please select a dataset first",
        variant: "destructive",
      });
      return;
    }
    optunaTuneMutation.mutate({ datasetId: selectedDataset, algorithm, nTrials: optunaTrials });
  };

  const latestModel = models?.filter(m => m.status === "completed").sort((a, b) =>
    new Date(b.trainedAt).getTime() - new Date(a.trainedAt).getTime()
  )[0];

  // Prepare trial data for visualization
  const trialData = trials?.map(t => ({
    trial: t.trialNumber,
    accuracy: t.value * 100,
  })) || [];

  const bestTrial = trials?.reduce((best, t) => t.value > (best?.value || 0) ? t : best, trials[0]);

  return (
    <div className="space-y-8">
      <Tabs defaultValue="baseline" className="w-full">
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="baseline" data-testid="tab-baseline">Baseline Training</TabsTrigger>
          <TabsTrigger value="optuna" data-testid="tab-optuna">Optuna Tuner</TabsTrigger>
        </TabsList>

        {/* Baseline Training Tab */}
        <TabsContent value="baseline" className="space-y-6 mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold">Training Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="dataset-select">Dataset</Label>
                  <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                    <SelectTrigger id="dataset-select" data-testid="select-dataset">
                      <SelectValue placeholder="Select dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets?.map((ds) => (
                        <SelectItem key={ds.id} value={ds.id}>
                          {ds.name} ({ds.rowCount} rows)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="algorithm-select">Algorithm</Label>
                  <Select value={algorithm} onValueChange={setAlgorithm}>
                    <SelectTrigger id="algorithm-select" data-testid="select-algorithm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="LogisticRegression">Logistic Regression</SelectItem>
                      <SelectItem value="RandomForest">Random Forest</SelectItem>
                      <SelectItem value="GradientBoosting">Gradient Boosting</SelectItem>
                      <SelectItem value="SVM">Support Vector Machine</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  onClick={handleBaselineTrain}
                  disabled={!selectedDataset || baselineTrainMutation.isPending}
                  className="w-full"
                  data-testid="button-train-baseline"
                >
                  <Play className="w-4 h-4 mr-2" />
                  {baselineTrainMutation.isPending ? "Training..." : "Train Baseline Model"}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold">Latest Model Performance</CardTitle>
              </CardHeader>
              <CardContent>
                {latestModel ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Model Type</span>
                      <Badge variant="secondary" className="font-mono">{latestModel.algorithm}</Badge>
                    </div>

                    <div className="space-y-3">
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Accuracy</span>
                          <span className="font-semibold text-foreground">
                            {(latestModel.metrics.accuracy * 100).toFixed(2)}%
                          </span>
                        </div>
                        <Progress value={latestModel.metrics.accuracy * 100} className="h-2" />
                      </div>

                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Precision</span>
                          <span className="font-semibold text-foreground">
                            {(latestModel.metrics.precision * 100).toFixed(2)}%
                          </span>
                        </div>
                        <Progress value={latestModel.metrics.precision * 100} className="h-2" />
                      </div>

                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Recall</span>
                          <span className="font-semibold text-foreground">
                            {(latestModel.metrics.recall * 100).toFixed(2)}%
                          </span>
                        </div>
                        <Progress value={latestModel.metrics.recall * 100} className="h-2" />
                      </div>

                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">F1 Score</span>
                          <span className="font-semibold text-foreground">
                            {(latestModel.metrics.f1Score * 100).toFixed(2)}%
                          </span>
                        </div>
                        <Progress value={latestModel.metrics.f1Score * 100} className="h-2" />
                      </div>
                    </div>

                    <div className="pt-3 border-t border-border text-xs text-muted-foreground font-mono">
                      Trained {new Date(latestModel.trainedAt).toLocaleString()}
                    </div>
                  </div>
                ) : (
                  <div className="h-[280px] flex flex-col items-center justify-center text-muted-foreground">
                    <Target className="w-12 h-12 mb-3 opacity-50" />
                    <div className="text-sm">No model trained yet</div>
                    <div className="text-xs mt-1">Train a baseline model to see metrics</div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Optuna Tuner Tab */}
        <TabsContent value="optuna" className="space-y-6 mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold">Tuning Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="optuna-dataset">Dataset</Label>
                  <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                    <SelectTrigger id="optuna-dataset" data-testid="select-optuna-dataset">
                      <SelectValue placeholder="Select dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets?.map((ds) => (
                        <SelectItem key={ds.id} value={ds.id}>
                          {ds.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="optuna-algorithm">Algorithm</Label>
                  <Select value={algorithm} onValueChange={setAlgorithm}>
                    <SelectTrigger id="optuna-algorithm" data-testid="select-optuna-algorithm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="LogisticRegression">Logistic Regression</SelectItem>
                      <SelectItem value="RandomForest">Random Forest</SelectItem>
                      <SelectItem value="GradientBoosting">Gradient Boosting</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="trials-count">Number of Trials</Label>
                  <Input
                    id="trials-count"
                    type="number"
                    min="5"
                    max="100"
                    value={optunaTrials}
                    onChange={(e) => setOptunaTrials(parseInt(e.target.value) || 20)}
                    data-testid="input-optuna-trials"
                  />
                </div>

                <Button
                  onClick={handleOptunaTune}
                  disabled={!selectedDataset || optunaTuneMutation.isPending}
                  className="w-full"
                  data-testid="button-run-optuna"
                >
                  <Zap className="w-4 h-4 mr-2" />
                  {optunaTuneMutation.isPending ? "Tuning..." : "Run Hyperparameter Tuning"}
                </Button>
              </CardContent>
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="text-lg font-semibold">Optimization History</CardTitle>
              </CardHeader>
              <CardContent>
                {trialData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={trialData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                      <XAxis
                        dataKey="trial"
                        className="text-xs text-muted-foreground"
                        label={{ value: "Trial Number", position: "insideBottom", offset: -5 }}
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
                      <Bar dataKey="accuracy" fill="hsl(var(--chart-2))" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-[300px] flex flex-col items-center justify-center text-muted-foreground">
                    <Zap className="w-12 h-12 mb-3 opacity-50" />
                    <div className="text-sm">No tuning trials yet</div>
                    <div className="text-xs mt-1">Run hyperparameter tuning to see results</div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Best Parameters */}
          {bestTrial && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold">Best Parameters Found</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-start gap-4">
                  <div className="flex-1">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(bestTrial.params).map(([key, value]) => (
                        <div key={key} className="space-y-1">
                          <div className="text-xs text-muted-foreground">{key}</div>
                          <div className="text-sm font-semibold font-mono text-foreground">
                            {typeof value === "number" ? value.toFixed(4) : String(value)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-muted-foreground">Best Accuracy</div>
                    <div className="text-2xl font-bold text-foreground">
                      {(bestTrial.value * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
