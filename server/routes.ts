import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import { randomUUID } from "crypto";

const upload = multer({ dest: "uploads/" });

export async function registerRoutes(app: Express): Promise<Server> {
  // ============ Dataset Routes ============
  app.get("/api/datasets", async (_req, res) => {
    const datasets = await storage.getDatasets();
    res.json(datasets);
  });

  app.get("/api/datasets/:id", async (req, res) => {
    const dataset = await storage.getDataset(req.params.id);
    if (!dataset) {
      return res.status(404).json({ error: "Dataset not found" });
    }
    res.json(dataset);
  });

  app.post("/api/datasets/upload", upload.single("file"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      const { name } = req.body;
      const fs = await import("fs/promises");
      const fileContent = await fs.readFile(req.file.path, "utf-8");
      
      // Parse CSV to get row and column counts
      const lines = fileContent.trim().split("\n");
      const headers = lines[0].split(",").map(h => h.trim());
      const rowCount = lines.length - 1; // Exclude header
      const columnCount = headers.length;

      const dataset = await storage.createDataset({
        name,
        filename: req.file.originalname,
        rowCount,
        columnCount,
        columns: headers,
      });

      await storage.createLog({
        level: "info",
        component: "dataset",
        message: `Dataset "${name}" uploaded successfully`,
        metadata: { datasetId: dataset.id, rows: rowCount, columns: columnCount },
      });

      // Clean up uploaded file
      await fs.unlink(req.file.path);

      res.json(dataset);
    } catch (error) {
      console.error("Upload error:", error);
      res.status(500).json({ error: "Failed to upload dataset" });
    }
  });

  // ============ Model Routes ============
  app.get("/api/models", async (_req, res) => {
    const models = await storage.getModels();
    res.json(models);
  });

  app.get("/api/models/:id", async (req, res) => {
    const model = await storage.getModel(req.params.id);
    if (!model) {
      return res.status(404).json({ error: "Model not found" });
    }
    res.json(model);
  });

  // ============ Training Routes ============
  app.post("/api/training/baseline", async (req, res) => {
    try {
      const { datasetId, algorithm } = req.body;

      if (!datasetId || !algorithm) {
        return res.status(400).json({ error: "Missing required fields" });
      }

      const dataset = await storage.getDataset(datasetId);
      if (!dataset) {
        return res.status(404).json({ error: "Dataset not found" });
      }

      await storage.createLog({
        level: "info",
        component: "training",
        message: `Starting baseline training with ${algorithm}`,
        metadata: { datasetId, algorithm },
      });

      // Simulate training (in real app, this would call Python ML code)
      const simulatedMetrics = {
        accuracy: 0.75 + Math.random() * 0.15,
        precision: 0.72 + Math.random() * 0.18,
        recall: 0.70 + Math.random() * 0.20,
        f1Score: 0.73 + Math.random() * 0.17,
      };

      const model = await storage.createModel({
        name: `${algorithm}_baseline_${Date.now()}`,
        type: "baseline",
        datasetId,
        algorithm,
        hyperparameters: getDefaultHyperparameters(algorithm),
        metrics: simulatedMetrics,
        featureTransformations: ["StandardScaler", "OneHotEncoder"],
        status: "completed",
      });

      await storage.createLog({
        level: "info",
        component: "training",
        message: `Baseline model trained successfully`,
        metadata: {
          modelId: model.id,
          accuracy: simulatedMetrics.accuracy,
          algorithm,
        },
      });

      res.json(model);
    } catch (error) {
      console.error("Training error:", error);
      res.status(500).json({ error: "Training failed" });
    }
  });

  app.post("/api/training/optuna", async (req, res) => {
    try {
      const { datasetId, algorithm, nTrials } = req.body;

      if (!datasetId || !algorithm || !nTrials) {
        return res.status(400).json({ error: "Missing required fields" });
      }

      const dataset = await storage.getDataset(datasetId);
      if (!dataset) {
        return res.status(404).json({ error: "Dataset not found" });
      }

      await storage.createLog({
        level: "info",
        component: "optimizer",
        message: `Starting Optuna hyperparameter tuning with ${nTrials} trials`,
        metadata: { datasetId, algorithm, nTrials },
      });

      // Simulate hyperparameter optimization
      const trials: any[] = [];
      let bestAccuracy = 0;
      let bestParams = {};

      // Create model first (with placeholder metrics)
      const model = await storage.createModel({
        name: `${algorithm}_optuna_${Date.now()}`,
        type: "retrained",
        datasetId,
        algorithm,
        hyperparameters: {},
        metrics: {
          accuracy: 0,
          precision: 0,
          recall: 0,
          f1Score: 0,
        },
        featureTransformations: ["StandardScaler", "OneHotEncoder"],
        status: "training",
      });

      for (let i = 0; i < nTrials; i++) {
        const params = generateRandomHyperparameters(algorithm);
        const accuracy = 0.65 + Math.random() * 0.25; // Simulated accuracy

        const trial = await storage.createOptunaTrial({
          modelId: model.id,
          trialNumber: i + 1,
          params,
          value: accuracy,
          state: "completed",
        });

        trials.push(trial);

        if (accuracy > bestAccuracy) {
          bestAccuracy = accuracy;
          bestParams = params;
        }
      }

      // Update model with best parameters
      await storage.updateModel(model.id, {
        hyperparameters: bestParams,
        metrics: {
          accuracy: bestAccuracy,
          precision: bestAccuracy * 0.95,
          recall: bestAccuracy * 0.97,
          f1Score: bestAccuracy * 0.96,
        },
        status: "completed",
      });

      await storage.createLog({
        level: "info",
        component: "optimizer",
        message: `Optuna tuning completed. Best accuracy: ${(bestAccuracy * 100).toFixed(2)}%`,
        metadata: {
          modelId: model.id,
          bestAccuracy,
          nTrials,
        },
      });

      res.json({ model, trials });
    } catch (error) {
      console.error("Optuna error:", error);
      res.status(500).json({ error: "Hyperparameter tuning failed" });
    }
  });

  // ============ Controller Routes ============
  app.get("/api/controller/steps", async (_req, res) => {
    const steps = await storage.getControllerSteps();
    res.json(steps);
  });

  app.post("/api/controller/step", async (req, res) => {
    try {
      const { modelId } = req.body;

      if (!modelId) {
        return res.status(400).json({ error: "Missing modelId" });
      }

      const model = await storage.getModel(modelId);
      if (!model) {
        return res.status(404).json({ error: "Model not found" });
      }

      // Find or create a standalone episode for this model
      let episodes = await storage.getEpisodes();
      let standaloneEpisode = episodes.find(
        ep => ep.modelId === modelId && ep.status === "running" && ep.episodeNumber === 0
      );

      if (!standaloneEpisode) {
        standaloneEpisode = await storage.createEpisode({
          modelId,
          episodeNumber: 0, // Special episode number for standalone steps
          totalSteps: 0,
          totalReward: 0,
          finalAccuracy: model.metrics.accuracy,
          retrainCount: 0,
          completedAt: null,
          status: "running",
        });
      }

      // Simulate RL agent decision (in real app, this would use the uploaded PPO model)
      const state = {
        currentAccuracy: model.metrics.accuracy,
        stepsSinceRetrain: Math.floor(Math.random() * 50),
        performanceDrift: Math.random() * 0.1,
        datasetSize: 1000 + Math.floor(Math.random() * 9000),
      };

      // Simple heuristic decision (placeholder for RL model)
      const action = state.performanceDrift > 0.05 ? "retrain" :
                     state.stepsSinceRetrain > 30 ? "tune_hyperparameters" :
                     "wait";

      const reward = action === "retrain" ? 10 : action === "tune_hyperparameters" ? 5 : 1;

      const allSteps = await storage.getControllerSteps();
      const episodeSteps = allSteps.filter(s => s.episodeId === standaloneEpisode.id);

      const step = await storage.createControllerStep({
        episodeId: standaloneEpisode.id,
        stepNumber: episodeSteps.length + 1,
        state,
        action,
        reward,
        nextState: {
          ...state,
          stepsSinceRetrain: action === "retrain" ? 0 : state.stepsSinceRetrain + 1,
        },
      });

      // Update episode
      await storage.updateEpisode(standaloneEpisode.id, {
        totalSteps: episodeSteps.length + 1,
        totalReward: standaloneEpisode.totalReward + reward,
      });

      await storage.createLog({
        level: "info",
        component: "controller",
        message: `Controller step executed: ${action}`,
        metadata: { stepId: step.id, action, reward },
      });

      res.json(step);
    } catch (error) {
      console.error("Controller step error:", error);
      res.status(500).json({ error: "Controller step failed" });
    }
  });

  app.post("/api/controller/simulate", async (req, res) => {
    try {
      const { modelId, maxSteps, numEpisodes } = req.body;

      if (!modelId || !maxSteps || !numEpisodes) {
        return res.status(400).json({ error: "Missing required fields" });
      }

      const model = await storage.getModel(modelId);
      if (!model) {
        return res.status(404).json({ error: "Model not found" });
      }

      await storage.createLog({
        level: "info",
        component: "controller",
        message: `Starting simulation: ${numEpisodes} episodes, ${maxSteps} steps each`,
        metadata: { modelId, maxSteps, numEpisodes },
      });

      const episodes: any[] = [];
      const allEpisodes = await storage.getEpisodes();
      const modelEpisodes = allEpisodes.filter(e => e.modelId === modelId && e.episodeNumber > 0);
      const nextEpisodeNumber = modelEpisodes.length > 0 ? 
        Math.max(...modelEpisodes.map(e => e.episodeNumber)) + 1 : 1;

      for (let ep = 0; ep < numEpisodes; ep++) {
        let totalReward = 0;
        let retrainCount = 0;
        let currentAccuracy = model.metrics.accuracy;

        const episode = await storage.createEpisode({
          modelId,
          episodeNumber: nextEpisodeNumber + ep,
          totalSteps: maxSteps,
          totalReward: 0,
          finalAccuracy: 0,
          retrainCount: 0,
          completedAt: null,
          status: "running",
        });

        for (let step = 1; step <= maxSteps; step++) {
          const state = {
            currentAccuracy,
            stepsSinceRetrain: step - (retrainCount * 10),
            performanceDrift: Math.random() * 0.1,
            datasetSize: 1000 + Math.floor(Math.random() * 9000),
          };

          const action = state.performanceDrift > 0.05 ? "retrain" :
                         (step - retrainCount * 10) > 30 ? "tune_hyperparameters" :
                         "wait";

          const reward = action === "retrain" ? 10 : action === "tune_hyperparameters" ? 5 : 1;
          totalReward += reward;

          if (action === "retrain") {
            retrainCount++;
            currentAccuracy = Math.min(0.95, currentAccuracy + 0.02);
          }

          await storage.createControllerStep({
            episodeId: episode.id,
            stepNumber: step,
            state,
            action,
            reward,
            nextState: {
              ...state,
              currentAccuracy,
              stepsSinceRetrain: action === "retrain" ? 0 : state.stepsSinceRetrain + 1,
            },
          });
        }

        await storage.updateEpisode(episode.id, {
          totalReward,
          finalAccuracy: currentAccuracy,
          retrainCount,
          completedAt: new Date(),
          status: "completed",
        });

        episodes.push(episode);
      }

      await storage.createLog({
        level: "info",
        component: "controller",
        message: `Simulation completed: ${numEpisodes} episodes finished`,
        metadata: { modelId, episodesCompleted: numEpisodes },
      });

      res.json({ episodes });
    } catch (error) {
      console.error("Simulation error:", error);
      res.status(500).json({ error: "Simulation failed" });
    }
  });

  // ============ Episode Routes ============
  app.get("/api/episodes", async (_req, res) => {
    const episodes = await storage.getEpisodes();
    res.json(episodes);
  });

  // ============ Log Routes ============
  app.get("/api/logs", async (_req, res) => {
    const logs = await storage.getLogs();
    res.json(logs);
  });

  // ============ Optuna Routes ============
  app.get("/api/optuna/trials", async (_req, res) => {
    const trials = await storage.getOptunaTrials();
    res.json(trials);
  });

  const httpServer = createServer(app);
  return httpServer;
}

// Helper functions
function getDefaultHyperparameters(algorithm: string): Record<string, any> {
  switch (algorithm) {
    case "LogisticRegression":
      return { C: 1.0, penalty: "l2", solver: "lbfgs", max_iter: 100 };
    case "RandomForest":
      return { n_estimators: 100, max_depth: 10, min_samples_split: 2 };
    case "GradientBoosting":
      return { n_estimators: 100, learning_rate: 0.1, max_depth: 3 };
    case "SVM":
      return { C: 1.0, kernel: "rbf", gamma: "scale" };
    default:
      return {};
  }
}

function generateRandomHyperparameters(algorithm: string): Record<string, any> {
  switch (algorithm) {
    case "LogisticRegression":
      return {
        C: Math.random() * 10,
        penalty: Math.random() > 0.5 ? "l1" : "l2",
        solver: ["lbfgs", "liblinear", "saga"][Math.floor(Math.random() * 3)],
        max_iter: 50 + Math.floor(Math.random() * 150),
      };
    case "RandomForest":
      return {
        n_estimators: 50 + Math.floor(Math.random() * 150),
        max_depth: 5 + Math.floor(Math.random() * 15),
        min_samples_split: 2 + Math.floor(Math.random() * 8),
        min_samples_leaf: 1 + Math.floor(Math.random() * 4),
      };
    case "GradientBoosting":
      return {
        n_estimators: 50 + Math.floor(Math.random() * 150),
        learning_rate: 0.01 + Math.random() * 0.29,
        max_depth: 2 + Math.floor(Math.random() * 6),
        subsample: 0.6 + Math.random() * 0.4,
      };
    default:
      return {};
  }
}
