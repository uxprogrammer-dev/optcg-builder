import { Injectable, Logger } from '@nestjs/common';

export interface ProgressUpdate {
  id: string;
  phase: 'leaders' | 'deck';
  step: string;
  progress: number; // 0-100
  details?: string;
  timestamp: number;
}

@Injectable()
export class ProgressService {
  private readonly logger = new Logger(ProgressService.name);
  private readonly progressStore = new Map<string, ProgressUpdate>();
  private readonly TTL = 5 * 60 * 1000; // 5 minutes

  /**
   * Create a new progress tracker
   */
  createProgress(id: string, phase: 'leaders' | 'deck'): void {
    this.progressStore.set(id, {
      id,
      phase,
      step: 'Initializing...',
      progress: 0,
      timestamp: Date.now(),
    });
    this.logger.debug(`Created progress tracker: ${id} (phase: ${phase})`);
  }

  /**
   * Update progress
   */
  updateProgress(
    id: string,
    step: string,
    progress: number,
    details?: string,
  ): void {
    const existing = this.progressStore.get(id);
    if (!existing) {
      this.logger.warn(`Progress tracker not found: ${id}`);
      return;
    }

    const update: ProgressUpdate = {
      ...existing,
      step,
      progress: Math.min(100, Math.max(0, progress)),
      details,
      timestamp: Date.now(),
    };

    this.progressStore.set(id, update);
    this.logger.log(`Progress update [${id}]: ${step} (${progress}%)`);
  }

  /**
   * Get current progress
   */
  getProgress(id: string): ProgressUpdate | null {
    const progress = this.progressStore.get(id);
    if (!progress) {
      return null;
    }

    // Check TTL
    if (Date.now() - progress.timestamp > this.TTL) {
      this.progressStore.delete(id);
      return null;
    }

    return progress;
  }

  /**
   * Complete progress
   */
  completeProgress(id: string, message: string = 'Complete'): void {
    this.updateProgress(id, message, 100);
    this.logger.debug(`Progress completed for ${id}: ${message}`);
    // Clean up after a delay - keep longer so frontend can poll
    setTimeout(() => {
      this.progressStore.delete(id);
      this.logger.debug(`Cleaned up completed progress: ${id}`);
    }, 30000); // Keep for 30 seconds after completion to allow frontend polling
  }

  /**
   * Fail progress
   */
  failProgress(id: string, message: string): void {
    this.updateProgress(id, `Error: ${message}`, 100);
    // Clean up after a delay
    setTimeout(() => {
      this.progressStore.delete(id);
    }, 30000); // Keep for 30 seconds after failure
  }

  /**
   * Clean up old progress entries
   */
  cleanup(): void {
    const now = Date.now();
    for (const [id, progress] of this.progressStore.entries()) {
      if (now - progress.timestamp > this.TTL) {
        this.progressStore.delete(id);
        this.logger.debug(`Cleaned up expired progress: ${id}`);
      }
    }
  }
}

