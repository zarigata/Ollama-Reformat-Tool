// Training configuration that matches the backend
// This type should be kept in sync with the TrainingConfig class in the backend
export interface TrainingConfig {
  // Basic parameters
  base_model: string;
  learning_rate: number;
  batch_size: number;
  num_epochs: number;
  
  // Model architecture
  context_length: number;
  lora_rank: number;
  lora_alpha: number;
  lora_dropout: number;
  
  // Training options
  use_unlock_prompt: boolean;
  custom_prompt_template?: string;
  train_test_split: number;
  early_stopping_patience: number;
  warmup_steps: number;
}

// Training status and progress
export interface TrainingStatus {
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number; // 0-100
  current_epoch?: number;
  total_epochs?: number;
  current_step?: number;
  total_steps?: number;
  metrics?: {
    loss?: number;
    learning_rate?: number;
    epoch?: number;
    step?: number;
    [key: string]: any;
  };
  error?: string;
  start_time?: string;
  end_time?: string;
  model_path?: string;
}

// Training job information
export interface TrainingJob {
  id: string;
  config: TrainingConfig;
  status: TrainingStatus;
  created_at: string;
  updated_at: string;
  created_by?: string;
  tags?: string[];
  description?: string;
}

// Model evaluation metrics
export interface ModelEvaluation {
  model_id: string;
  job_id: string;
  metrics: {
    loss: number;
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    perplexity?: number;
    [key: string]: any;
  };
  test_results?: {
    input: string;
    expected: string;
    predicted: string;
    is_correct: boolean;
    confidence?: number;
  }[];
  created_at: string;
}

// Model information
export interface TrainedModel {
  id: string;
  name: string;
  description?: string;
  base_model: string;
  path: string;
  size_mb: number;
  created_at: string;
  updated_at: string;
  training_job_id?: string;
  metrics?: {
    training_loss?: number;
    validation_loss?: number;
    [key: string]: any;
  };
  tags?: string[];
  is_public: boolean;
  created_by?: string;
}
