import { Router } from 'express';
import { runAdvisorAgent, runStructuredAdvisorAgent } from '../services/aiAdvisor.js';

const router = Router();

/**
 * POST /api/advisor/chat
 * Body: { message: string }
 * Streams SSE events with agent steps and final recommendation.
 */
router.post('/advisor/chat', async (req, res) => {
  const { message } = req.body;

  if (!message || typeof message !== 'string' || message.trim().length < 5) {
    return res.status(400).json({ error: 'Please provide a valid message describing your investment goals.' });
  }

  // Set up Server-Sent Events
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  try {
    await runAdvisorAgent(message.trim(), res);
  } catch (err) {
    console.error('[Advisor Route] Unexpected error:', err);
    res.write(`data: ${JSON.stringify({ type: 'error', message: 'An unexpected error occurred.' })}\n\n`);
  } finally {
    res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
    res.end();
  }
});

/**
 * POST /api/advisor/plan
 * Body: { params: object }
 * Skips extraction step and directly screens funds and generates AI response.
 */
router.post('/advisor/plan', async (req, res) => {
  const { params } = req.body;

  if (!params || typeof params !== 'object') {
    return res.status(400).json({ error: 'Missing structured parameters.' });
  }

  // Set up Server-Sent Events
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  try {
    await runStructuredAdvisorAgent(params, res);
  } catch (err) {
    console.error('[Advisor Route] Unexpected error in plan:', err);
    res.write(`data: ${JSON.stringify({ type: 'error', message: 'An unexpected error occurred.' })}\n\n`);
  } finally {
    res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
    res.end();
  }
});

/**
 * POST /api/advisor/analyze
 * Body: { holdings: array }
 * Runs AI Portfolio Analyzer.
 */
router.post('/advisor/analyze', async (req, res) => {
  const { holdings } = req.body;

  if (!holdings || !Array.isArray(holdings) || holdings.length === 0) {
    return res.status(400).json({ error: 'Missing or invalid holdings data.' });
  }

  // Set up Server-Sent Events
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  try {
    const { runAnalyzerAgent } = await import('../services/aiAdvisor.js');
    await runAnalyzerAgent(holdings, res);
  } catch (err) {
    console.error('[Advisor Route] Unexpected error in analyze:', err);
    res.write(`data: ${JSON.stringify({ type: 'error', message: 'An unexpected error occurred during analysis.' })}\n\n`);
  } finally {
    res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
    res.end();
  }
});

export default router;
