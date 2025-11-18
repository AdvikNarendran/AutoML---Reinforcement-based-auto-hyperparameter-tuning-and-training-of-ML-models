import {
  type Dataset,
  type InsertDataset,
  type Model,
  type InsertModel,
  type Episode,
  type InsertEpisode,
  type ControllerStep,
  type InsertControllerStep,
  type Log,
  type InsertLog,
  type OptunaTrial,
  type InsertOptunaTrial,
} from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  // Dataset methods
  getDatasets(): Promise<Dataset[]>;
  getDataset(id: string): Promise<Dataset | undefined>;
  createDataset(dataset: InsertDataset): Promise<Dataset>;
  deleteDataset(id: string): Promise<void>;

  // Model methods
  getModels(): Promise<Model[]>;
  getModel(id: string): Promise<Model | undefined>;
  createModel(model: InsertModel): Promise<Model>;
  updateModel(id: string, updates: Partial<Model>): Promise<Model | undefined>;

  // Episode methods
  getEpisodes(): Promise<Episode[]>;
  getEpisode(id: string): Promise<Episode | undefined>;
  createEpisode(episode: InsertEpisode): Promise<Episode>;
  updateEpisode(id: string, updates: Partial<Episode>): Promise<Episode | undefined>;

  // Controller Step methods
  getControllerSteps(): Promise<ControllerStep[]>;
  getControllerStepsByEpisode(episodeId: string): Promise<ControllerStep[]>;
  createControllerStep(step: InsertControllerStep): Promise<ControllerStep>;

  // Log methods
  getLogs(): Promise<Log[]>;
  createLog(log: InsertLog): Promise<Log>;

  // Optuna Trial methods
  getOptunaTrials(): Promise<OptunaTrial[]>;
  getOptunaTrialsByModel(modelId: string): Promise<OptunaTrial[]>;
  createOptunaTrial(trial: InsertOptunaTrial): Promise<OptunaTrial>;
}

export class MemStorage implements IStorage {
  private datasets: Map<string, Dataset>;
  private models: Map<string, Model>;
  private episodes: Map<string, Episode>;
  private controllerSteps: Map<string, ControllerStep>;
  private logs: Map<string, Log>;
  private optunaTrials: Map<string, OptunaTrial>;

  constructor() {
    this.datasets = new Map();
    this.models = new Map();
    this.episodes = new Map();
    this.controllerSteps = new Map();
    this.logs = new Map();
    this.optunaTrials = new Map();
  }

  // Dataset methods
  async getDatasets(): Promise<Dataset[]> {
    return Array.from(this.datasets.values());
  }

  async getDataset(id: string): Promise<Dataset | undefined> {
    return this.datasets.get(id);
  }

  async createDataset(insertDataset: InsertDataset): Promise<Dataset> {
    const id = randomUUID();
    const dataset: Dataset = {
      ...insertDataset,
      id,
      uploadedAt: new Date(),
    };
    this.datasets.set(id, dataset);
    return dataset;
  }

  async deleteDataset(id: string): Promise<void> {
    this.datasets.delete(id);
  }

  // Model methods
  async getModels(): Promise<Model[]> {
    return Array.from(this.models.values());
  }

  async getModel(id: string): Promise<Model | undefined> {
    return this.models.get(id);
  }

  async createModel(insertModel: InsertModel): Promise<Model> {
    const id = randomUUID();
    const model: Model = {
      ...insertModel,
      id,
      trainedAt: new Date(),
    };
    this.models.set(id, model);
    return model;
  }

  async updateModel(id: string, updates: Partial<Model>): Promise<Model | undefined> {
    const model = this.models.get(id);
    if (!model) return undefined;
    const updated = { ...model, ...updates };
    this.models.set(id, updated);
    return updated;
  }

  // Episode methods
  async getEpisodes(): Promise<Episode[]> {
    return Array.from(this.episodes.values());
  }

  async getEpisode(id: string): Promise<Episode | undefined> {
    return this.episodes.get(id);
  }

  async createEpisode(insertEpisode: InsertEpisode): Promise<Episode> {
    const id = randomUUID();
    const episode: Episode = {
      ...insertEpisode,
      id,
      startedAt: new Date(),
    };
    this.episodes.set(id, episode);
    return episode;
  }

  async updateEpisode(id: string, updates: Partial<Episode>): Promise<Episode | undefined> {
    const episode = this.episodes.get(id);
    if (!episode) return undefined;
    const updated = { ...episode, ...updates };
    this.episodes.set(id, updated);
    return updated;
  }

  // Controller Step methods
  async getControllerSteps(): Promise<ControllerStep[]> {
    return Array.from(this.controllerSteps.values());
  }

  async getControllerStepsByEpisode(episodeId: string): Promise<ControllerStep[]> {
    return Array.from(this.controllerSteps.values()).filter(
      (step) => step.episodeId === episodeId
    );
  }

  async createControllerStep(insertStep: InsertControllerStep): Promise<ControllerStep> {
    const id = randomUUID();
    const step: ControllerStep = {
      ...insertStep,
      id,
      timestamp: new Date(),
    };
    this.controllerSteps.set(id, step);
    return step;
  }

  // Log methods
  async getLogs(): Promise<Log[]> {
    return Array.from(this.logs.values());
  }

  async createLog(insertLog: InsertLog): Promise<Log> {
    const id = randomUUID();
    const log: Log = {
      ...insertLog,
      id,
      timestamp: new Date(),
    };
    this.logs.set(id, log);
    return log;
  }

  // Optuna Trial methods
  async getOptunaTrials(): Promise<OptunaTrial[]> {
    return Array.from(this.optunaTrials.values());
  }

  async getOptunaTrialsByModel(modelId: string): Promise<OptunaTrial[]> {
    return Array.from(this.optunaTrials.values()).filter(
      (trial) => trial.modelId === modelId
    );
  }

  async createOptunaTrial(insertTrial: InsertOptunaTrial): Promise<OptunaTrial> {
    const id = randomUUID();
    const trial: OptunaTrial = {
      ...insertTrial,
      id,
      timestamp: new Date(),
    };
    this.optunaTrials.set(id, trial);
    return trial;
  }
}

export const storage = new MemStorage();
