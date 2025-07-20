import { useEffect, useState } from 'react';
import { TrainingStatus } from '../types/training';

interface TrainingStatusProps {
  jobId: string;
  onComplete?: (modelPath: string) => void;
  onError?: (error: string) => void;
  refreshInterval?: number;
}

const POLLING_INTERVAL = 5000; // 5 seconds

export default function TrainingStatus({
  jobId,
  onComplete,
  onError,
  refreshInterval = POLLING_INTERVAL,
}: TrainingStatusProps) {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const response = await fetch(`/api/training/status/${jobId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setStatus(data);
      setLoading(false);

      // Handle completion
      if (data.status === 'completed' && onComplete && data.model_path) {
        onComplete(data.model_path);
      }

      // Handle errors
      if (data.status === 'failed' && onError) {
        onError(data.error || 'Training failed');
      }

      return data;
    } catch (err) {
      console.error('Error fetching training status:', err);
      setError('Failed to fetch training status');
      setLoading(false);
      return null;
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchStatus();

    // Set up polling
    const intervalId = setInterval(() => {
      if (status?.status === 'completed' || status?.status === 'failed') {
        clearInterval(intervalId);
        return;
      }
      fetchStatus();
    }, refreshInterval);

    // Clean up
    return () => clearInterval(intervalId);
  }, [jobId, refreshInterval]);

  if (loading) {
    return (
      <div className="text-center py-4">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
        <p className="mt-2 text-sm text-gray-600">Checking training status...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border-l-4 border-red-400 p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-sm text-yellow-700">No status information available</p>
          </div>
        </div>
      </div>
    );
  }

  const progress = Math.min(100, Math.max(0, status.progress || 0));
  const isComplete = status.status === 'completed';
  const isFailed = status.status === 'failed';
  const isRunning = status.status === 'running';

  return (
    <div className="space-y-4">
      <div>
        <div className="flex justify-between text-sm font-medium mb-1">
          <span>
            {isComplete ? 'Completed' : isFailed ? 'Failed' : isRunning ? 'Training in progress' : 'Pending'}
          </span>
          <span>{progress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className={`h-2.5 rounded-full ${
              isComplete ? 'bg-green-500' : 
              isFailed ? 'bg-red-500' : 
              'bg-blue-600'}`}
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div>
          <div className="text-gray-500">Epoch</div>
          <div className="font-medium">
            {status.current_epoch !== undefined && status.total_epochs !== undefined
              ? `${status.current_epoch} / ${status.total_epochs}`
              : 'N/A'}
          </div>
        </div>
        <div>
          <div className="text-gray-500">Step</div>
          <div className="font-medium">
            {status.current_step !== undefined && status.total_steps !== undefined
              ? `${status.current_step} / ${status.total_steps}`
              : 'N/A'}
          </div>
        </div>
        {status.metrics?.loss !== undefined && (
          <div>
            <div className="text-gray-500">Loss</div>
            <div className="font-mono">{status.metrics.loss.toFixed(4)}</div>
          </div>
        )}
        {status.metrics?.learning_rate !== undefined && (
          <div>
            <div className="text-gray-500">Learning Rate</div>
            <div className="font-mono">{status.metrics.learning_rate.toExponential(2)}</div>
          </div>
        )}
      </div>

      {status.error && (
        <div className="mt-4 p-3 bg-red-50 border-l-4 border-red-400">
          <div className="text-sm text-red-700">
            <p className="font-medium">Error:</p>
            <p className="mt-1">{status.error}</p>
          </div>
        </div>
      )}

      {isComplete && status.model_path && (
        <div className="mt-4 p-3 bg-green-50 border-l-4 border-green-400">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-green-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-green-800">
                Training completed successfully!
              </p>
              <p className="text-sm text-green-700 mt-1">
                Model saved to: <code className="bg-green-100 px-1 rounded">{status.model_path}</code>
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
