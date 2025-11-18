import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, Pause, SkipForward, SkipBack, RefreshCw, Brain } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { Model, ControllerStep, Episode } from "@shared/schema";

export default function Controller() {
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [maxSteps, setMaxSteps] = useState(50);
  const [numEpisodes, setNumEpisodes] = useState(5);
  const { toast } = useToast();

  const { data: models } = useQuery<Model[]>({
    queryKey: ["/api/models"],
  });

  const { data: steps } = useQuery<ControllerStep[]>({
    queryKey: ["/api/controller/steps"],
    enabled: !!selectedModel,
  });

  const { data: episodes } = useQuery<Episode[]>({
    queryKey: ["/api/episodes"],
  });

  const stepMutation = useMutation({
    mutationFn: async (data: { modelId: string }) =>
      apiRequest("POST", "/api/controller/step", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/controller/steps"] });
      queryClient.invalidateQueries({ queryKey: ["/api/logs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/episodes"] });
      toast({
        title: "Controller step executed",
        description: "RL agent decision applied",
      });
      if (steps && steps.length > 0) {
        setCurrentStepIndex(steps.length);
      }
    },
    onError: () => {
      toast({
        title: "Step failed",
        description: "There was an error executing the controller step",
        variant: "destructive",
      });
    },
  });

  const simulateMutation = useMutation({
    mutationFn: async (data: { modelId: string; maxSteps: number; numEpisodes: number }) =>
      apiRequest("POST", "/api/controller/simulate", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/episodes"] });
      queryClient.invalidateQueries({ queryKey: ["/api/controller/steps"] });
      queryClient.invalidateQueries({ queryKey: ["/api/logs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/models"] });
      toast({
        title: "Simulation completed",
        description: "Episodes have been executed successfully",
      });
    },
    onError: () => {
      toast({
        title: "Simulation failed",
        description: "There was an error running the simulation",
        variant: "destructive",
      });
    },
  });

  const currentStep = steps?.[currentStepIndex];
  const completedModels = models?.filter(m => m.status === "completed") || [];

  const handleStepForward = () => {
    if (!selectedModel) {
      toast({
        title: "No model selected",
        description: "Please select a model first",
        variant: "destructive",
      });
      return;
    }
    stepMutation.mutate({ modelId: selectedModel });
  };

  const handleSimulate = () => {
    if (!selectedModel) {
      toast({
        title: "No model selected",
        description: "Please select a model first",
        variant: "destructive",
      });
      return;
    }
    simulateMutation.mutate({ modelId: selectedModel, maxSteps, numEpisodes });
  };

  return (
    <div className="space-y-8">
      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Controller Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="controller-model">Base Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger id="controller-model" data-testid="select-controller-model">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {completedModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name} ({(model.metrics.accuracy * 100).toFixed(1)}%)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="max-steps">Max Steps per Episode</Label>
              <Input
                id="max-steps"
                type="number"
                min="10"
                max="200"
                value={maxSteps}
                onChange={(e) => setMaxSteps(parseInt(e.target.value) || 50)}
                data-testid="input-max-steps"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="num-episodes">Number of Episodes</Label>
              <Input
                id="num-episodes"
                type="number"
                min="1"
                max="20"
                value={numEpisodes}
                onChange={(e) => setNumEpisodes(parseInt(e.target.value) || 5)}
                data-testid="input-num-episodes"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Step-by-Step Execution */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2">
          <CardTitle className="text-lg font-semibold">Step-by-Step Execution</CardTitle>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => setCurrentStepIndex(Math.max(0, currentStepIndex - 1))}
              disabled={currentStepIndex === 0 || !steps?.length}
              data-testid="button-step-back"
            >
              <SkipBack className="w-4 h-4" />
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={handleStepForward}
              disabled={!selectedModel || stepMutation.isPending}
              data-testid="button-step-forward"
            >
              <SkipForward className="w-4 h-4" />
            </Button>
            <Button
              size="sm"
              onClick={() => setCurrentStepIndex(Math.min((steps?.length || 1) - 1, currentStepIndex + 1))}
              disabled={!steps?.length || currentStepIndex >= (steps?.length || 0) - 1}
              data-testid="button-next-step"
            >
              <Play className="w-4 h-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {currentStep ? (
            <div className="space-y-6">
              {/* State Display */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-sm font-semibold text-foreground mb-3">Current State</h3>
                  <div className="space-y-2 p-4 rounded-md bg-muted/50 border border-border">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Accuracy</span>
                      <span className="text-sm font-mono font-semibold text-foreground">
                        {(currentStep.state.currentAccuracy * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Steps Since Retrain</span>
                      <span className="text-sm font-mono font-semibold text-foreground">
                        {currentStep.state.stepsSinceRetrain}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Performance Drift</span>
                      <span className="text-sm font-mono font-semibold text-foreground">
                        {currentStep.state.performanceDrift.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Dataset Size</span>
                      <span className="text-sm font-mono font-semibold text-foreground">
                        {currentStep.state.datasetSize.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-semibold text-foreground mb-3">RL Decision</h3>
                  <div className="p-6 rounded-md bg-primary/10 border border-primary/20 text-center">
                    <Brain className="w-12 h-12 mx-auto mb-3 text-primary" />
                    <Badge
                      variant={currentStep.action === "retrain" ? "default" : "secondary"}
                      className="text-sm font-mono mb-2"
                    >
                      {currentStep.action}
                    </Badge>
                    <div className="text-xs text-muted-foreground mt-3">
                      Reward: <span className="font-mono font-semibold">{currentStep.reward.toFixed(3)}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Step Timeline */}
              <div>
                <h3 className="text-sm font-semibold text-foreground mb-3">
                  Step {currentStep.stepNumber} Timeline
                </h3>
                <div className="flex items-center gap-2 overflow-x-auto pb-2">
                  {steps.slice(0, 20).map((step, idx) => (
                    <button
                      key={step.id}
                      onClick={() => setCurrentStepIndex(idx)}
                      className={`flex-shrink-0 w-8 h-8 rounded-md text-xs font-mono font-semibold transition-colors ${
                        idx === currentStepIndex
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground hover-elevate"
                      }`}
                      data-testid={`step-${idx}`}
                    >
                      {step.stepNumber}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-[300px] flex flex-col items-center justify-center text-muted-foreground">
              <Brain className="w-12 h-12 mb-3 opacity-50" />
              <div className="text-sm">No controller steps yet</div>
              <div className="text-xs mt-1">Execute a step or run a simulation</div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Simulate Episodes */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Simulate Episodes</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button
            onClick={handleSimulate}
            disabled={!selectedModel || simulateMutation.isPending}
            className="w-full"
            data-testid="button-simulate-episodes"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            {simulateMutation.isPending ? "Simulating..." : `Run ${numEpisodes} Episodes`}
          </Button>

          {/* Recent Episodes Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4">
            {episodes?.slice(-3).reverse().map((episode) => (
              <div
                key={episode.id}
                className="p-4 rounded-md border border-border hover-elevate"
                data-testid={`episode-summary-${episode.id}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold font-mono">Ep {episode.episodeNumber}</span>
                  <Badge variant={episode.status === "completed" ? "default" : "secondary"}>
                    {episode.status}
                  </Badge>
                </div>
                <div className="space-y-1 text-xs text-muted-foreground">
                  <div>Steps: {episode.totalSteps}</div>
                  <div>Retrains: {episode.retrainCount}</div>
                  <div>Reward: {episode.totalReward.toFixed(2)}</div>
                  <div>Final Acc: {(episode.finalAccuracy * 100).toFixed(1)}%</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
