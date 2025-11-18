import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, real, jsonb, timestamp, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Dataset schema
export const datasets = pgTable("datasets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  filename: text("filename").notNull(),
  rowCount: integer("row_count").notNull(),
  columnCount: integer("column_count").notNull(),
  columns: jsonb("columns").notNull().$type<string[]>(),
  uploadedAt: timestamp("uploaded_at").notNull().defaultNow(),
});

export const insertDatasetSchema = createInsertSchema(datasets).omit({
  id: true,
  uploadedAt: true,
});

export type InsertDataset = z.infer<typeof insertDatasetSchema>;
export type Dataset = typeof datasets.$inferSelect;

// Model schema
export const models = pgTable("models", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  type: text("type").notNull(), // "baseline" | "retrained"
  datasetId: varchar("dataset_id").notNull(),
  algorithm: text("algorithm").notNull(), // e.g., "LogisticRegression", "RandomForest"
  hyperparameters: jsonb("hyperparameters").notNull().$type<Record<string, any>>(),
  metrics: jsonb("metrics").notNull().$type<{
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  }>(),
  featureTransformations: jsonb("feature_transformations").$type<string[]>(),
  trainedAt: timestamp("trained_at").notNull().defaultNow(),
  status: text("status").notNull(), // "training" | "completed" | "failed"
});

export const insertModelSchema = createInsertSchema(models).omit({
  id: true,
  trainedAt: true,
});

export type InsertModel = z.infer<typeof insertModelSchema>;
export type Model = typeof models.$inferSelect;

// Episode schema
export const episodes = pgTable("episodes", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  modelId: varchar("model_id").notNull(),
  episodeNumber: integer("episode_number").notNull(),
  totalSteps: integer("total_steps").notNull(),
  totalReward: real("total_reward").notNull(),
  finalAccuracy: real("final_accuracy").notNull(),
  retrainCount: integer("retrain_count").notNull(),
  startedAt: timestamp("started_at").notNull().defaultNow(),
  completedAt: timestamp("completed_at"),
  status: text("status").notNull(), // "running" | "completed" | "failed"
});

export const insertEpisodeSchema = createInsertSchema(episodes).omit({
  id: true,
  startedAt: true,
});

export type InsertEpisode = z.infer<typeof insertEpisodeSchema>;
export type Episode = typeof episodes.$inferSelect;

// Controller Step schema
export const controllerSteps = pgTable("controller_steps", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  episodeId: varchar("episode_id").notNull(),
  stepNumber: integer("step_number").notNull(),
  state: jsonb("state").notNull().$type<{
    currentAccuracy: number;
    stepsSinceRetrain: number;
    performanceDrift: number;
    datasetSize: number;
  }>(),
  action: text("action").notNull(), // "retrain" | "wait" | "tune_hyperparameters"
  reward: real("reward").notNull(),
  nextState: jsonb("next_state").$type<{
    currentAccuracy: number;
    stepsSinceRetrain: number;
    performanceDrift: number;
    datasetSize: number;
  }>(),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
});

export const insertControllerStepSchema = createInsertSchema(controllerSteps).omit({
  id: true,
  timestamp: true,
});

export type InsertControllerStep = z.infer<typeof insertControllerStepSchema>;
export type ControllerStep = typeof controllerSteps.$inferSelect;

// Log schema
export const logs = pgTable("logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  level: text("level").notNull(), // "info" | "warning" | "error"
  component: text("component").notNull(), // "dataset" | "training" | "controller" | "optimizer" | "system"
  message: text("message").notNull(),
  metadata: jsonb("metadata").$type<Record<string, any>>(),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
});

export const insertLogSchema = createInsertSchema(logs).omit({
  id: true,
  timestamp: true,
});

export type InsertLog = z.infer<typeof insertLogSchema>;
export type Log = typeof logs.$inferSelect;

// Optuna Trial schema
export const optunaTri als = pgTable("optuna_trials", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  modelId: varchar("model_id").notNull(),
  trialNumber: integer("trial_number").notNull(),
  params: jsonb("params").notNull().$type<Record<string, any>>(),
  value: real("value").notNull(), // optimization metric (e.g., accuracy)
  state: text("state").notNull(), // "running" | "completed" | "failed"
  timestamp: timestamp("timestamp").notNull().defaultNow(),
});

export const insertOptunaTrialSchema = createInsertSchema(optunaTrials).omit({
  id: true,
  timestamp: true,
});

export type InsertOptunaTrial = z.infer<typeof insertOptunaTrialSchema>;
export type OptunaTrial = typeof optunaTrials.$inferSelect;
