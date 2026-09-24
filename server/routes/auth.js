import { Router } from 'express';
import { OAuth2Client } from 'google-auth-library';
import jwt from 'jsonwebtoken';
import { randomUUID } from 'crypto';
import { getDB } from '../db.js';

const router = Router();
const client = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);
const JWT_SECRET = process.env.JWT_SECRET || 'fundsense_jwt_secret_default_key_2026';

/**
 * POST /api/auth/google
 * Body: { credential }
 * Verifies the Google ID token and creates or updates the user in SQLite.
 */
router.post('/google', async (req, res) => {
  const { credential } = req.body;

  if (!credential) {
    return res.status(400).json({ error: 'Missing Google credential token.' });
  }

  try {
    const audience = process.env.GOOGLE_CLIENT_ID;
    if (!audience) {
      console.error('[Auth] GOOGLE_CLIENT_ID is not configured in environment.');
      return res.status(500).json({ error: 'Server authentication configuration missing.' });
    }

    const ticket = await client.verifyIdToken({
      idToken: credential,
      audience,
    });

    const payload = ticket.getPayload();
    if (!payload || !payload.sub || !payload.email) {
      return res.status(400).json({ error: 'Invalid Google token payload.' });
    }

    const { sub: googleId, email, name, picture: avatarUrl, given_name: givenName, family_name: familyName } = payload;
    const db = getDB();

    // Look up existing user by google_id or email
    let user = db.prepare('SELECT * FROM users WHERE google_id = ? OR email = ?').get(googleId, email);

    if (user) {
      // Update existing record
      db.prepare(`
        UPDATE users 
        SET email = ?, 
            google_id = ?,
            name = COALESCE(?, name), 
            first_name = COALESCE(first_name, ?),
            last_name = COALESCE(last_name, ?),
            avatar_url = COALESCE(?, avatar_url), 
            last_login_at = datetime('now')
        WHERE id = ?
      `).run(email, googleId, name, givenName || '', familyName || '', avatarUrl, user.id);

      user = db.prepare('SELECT * FROM users WHERE id = ?').get(user.id);
    } else {
      // Deterministic user ID ensures stability across ephemeral deployments
      const deterministicId = 'usr_' + Buffer.from(email.toLowerCase()).toString('hex').slice(0, 24);
      db.prepare(`
        INSERT INTO users (id, google_id, email, name, first_name, last_name, avatar_url, profile_completed, last_login_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          email = excluded.email,
          google_id = excluded.google_id,
          name = COALESCE(excluded.name, users.name),
          avatar_url = COALESCE(excluded.avatar_url, users.avatar_url),
          last_login_at = datetime('now')
      `).run(deterministicId, googleId, email, name || '', givenName || '', familyName || '', avatarUrl || '');

      user = db.prepare('SELECT * FROM users WHERE id = ?').get(deterministicId);
    }

    // Sign session JWT (valid for 30 days)
    const sessionToken = jwt.sign(
      {
        id: user.id,
        email: user.email,
        name: user.name,
        avatar_url: user.avatar_url
      },
      JWT_SECRET,
      { expiresIn: '30d' }
    );

    console.log(`[Auth] User authenticated successfully: ${user.email} (${user.id})`);

    return res.json({
      success: true,
      token: sessionToken,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        firstName: user.first_name,
        lastName: user.last_name,
        avatarUrl: user.avatar_url,
        age: user.age,
        profession: user.profession,
        investmentExperience: user.investment_experience,
        monthlyInvestmentBracket: user.monthly_investment_bracket,
        profileCompleted: Boolean(user.profile_completed || (user.profession && user.age)),
        createdAt: user.created_at
      }
    });

  } catch (err) {
    console.error('[Auth] Token verification failed:', err.message);
    return res.status(401).json({ error: 'Failed to verify Google token: ' + err.message });
  }
});

/**
 * PUT /api/auth/profile
 * Headers: { Authorization: 'Bearer <token>' }
 * Body: { firstName, lastName, age, profession, investmentExperience, monthlyInvestmentBracket }
 * Updates demographic and profile details.
 */
router.put('/profile', (req, res) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'No authorization token provided.' });
  }

  const token = authHeader.split(' ')[1];

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    const db = getDB();
    let user = db.prepare('SELECT * FROM users WHERE id = ?').get(decoded.id);

    if (!user && decoded.email) {
      user = db.prepare('SELECT * FROM users WHERE email = ?').get(decoded.email);
    }

    if (!user && decoded.id && decoded.email) {
      db.prepare(`
        INSERT INTO users (id, google_id, email, name, avatar_url, profile_completed, last_login_at)
        VALUES (?, ?, ?, ?, ?, 1, datetime('now'))
        ON CONFLICT(id) DO NOTHING
      `).run(decoded.id, decoded.id, decoded.email, decoded.name || '', decoded.avatar_url || '');
      user = db.prepare('SELECT * FROM users WHERE id = ?').get(decoded.id);
    }

    if (!user) {
      return res.status(404).json({ error: 'User not found.' });
    }

    const {
      firstName,
      lastName,
      age,
      profession,
      investmentExperience,
      monthlyInvestmentBracket
    } = req.body;

    const safeFirstName = typeof firstName === 'string' ? firstName.trim().slice(0, 50) : user.first_name;
    const safeLastName = typeof lastName === 'string' ? lastName.trim().slice(0, 50) : user.last_name;
    const fullName = [safeFirstName, safeLastName].filter(Boolean).join(' ') || user.name;

    let parsedAge = user.age;
    if (age !== undefined && age !== null && age !== '') {
      const parsed = parseInt(age, 10);
      if (!isNaN(parsed) && parsed >= 18 && parsed <= 120) {
        parsedAge = parsed;
      }
    }

    const safeProfession = typeof profession === 'string' ? profession.trim().slice(0, 100) : user.profession;
    const safeExperience = typeof investmentExperience === 'string' ? investmentExperience.trim().slice(0, 50) : user.investment_experience;
    const safeBracket = typeof monthlyInvestmentBracket === 'string' ? monthlyInvestmentBracket.trim().slice(0, 50) : user.monthly_investment_bracket;

    db.prepare(`
      UPDATE users
      SET first_name = ?,
          last_name = ?,
          name = ?,
          age = ?,
          profession = ?,
          investment_experience = ?,
          monthly_investment_bracket = ?,
          profile_completed = 1
      WHERE id = ?
    `).run(
      safeFirstName,
      safeLastName,
      fullName,
      parsedAge,
      safeProfession,
      safeExperience,
      safeBracket,
      user.id
    );

    const updatedUser = db.prepare('SELECT * FROM users WHERE id = ?').get(user.id);

    return res.json({
      success: true,
      user: {
        id: updatedUser.id,
        email: updatedUser.email,
        name: updatedUser.name,
        firstName: updatedUser.first_name,
        lastName: updatedUser.last_name,
        avatarUrl: updatedUser.avatar_url,
        age: updatedUser.age,
        profession: updatedUser.profession,
        investmentExperience: updatedUser.investment_experience,
        monthlyInvestmentBracket: updatedUser.monthly_investment_bracket,
        profileCompleted: Boolean(updatedUser.profile_completed),
        createdAt: updatedUser.created_at,
        lastLoginAt: updatedUser.last_login_at
      }
    });
  } catch (err) {
    console.error('[Auth] Profile update error:', err);
    return res.status(401).json({ error: 'Session expired or invalid token.' });
  }
});

/**
 * GET /api/auth/me
 * Headers: { Authorization: 'Bearer <token>' }
 * Returns the currently authenticated user profile.
 */
router.get('/me', (req, res) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'No authorization token provided.' });
  }

  const token = authHeader.split(' ')[1];

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    const db = getDB();
    let user = db.prepare('SELECT * FROM users WHERE id = ?').get(decoded.id);

    if (!user && decoded.email) {
      user = db.prepare('SELECT * FROM users WHERE email = ?').get(decoded.email);
    }

    if (!user && decoded.id && decoded.email) {
      // Ephemeral server restart: auto-restore user record from verified JWT session
      const givenName = decoded.name ? decoded.name.split(' ')[0] : '';
      const familyName = decoded.name ? decoded.name.split(' ').slice(1).join(' ') : '';
      db.prepare(`
        INSERT INTO users (id, google_id, email, name, first_name, last_name, avatar_url, profile_completed, last_login_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          email = excluded.email,
          name = COALESCE(excluded.name, users.name),
          last_login_at = datetime('now')
      `).run(decoded.id, decoded.id, decoded.email, decoded.name || '', givenName, familyName, decoded.avatar_url || '');

      user = db.prepare('SELECT * FROM users WHERE id = ?').get(decoded.id);
    }

    if (!user) {
      return res.status(401).json({ error: 'User session invalid or expired.' });
    }

    return res.json({
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        firstName: user.first_name,
        lastName: user.last_name,
        avatarUrl: user.avatar_url,
        age: user.age,
        profession: user.profession,
        investmentExperience: user.investment_experience,
        monthlyInvestmentBracket: user.monthly_investment_bracket,
        profileCompleted: Boolean(user.profile_completed || (user.profession && user.age)),
        createdAt: user.created_at,
        lastLoginAt: user.last_login_at
      }
    });
  } catch (err) {
    return res.status(401).json({ error: 'Session expired or invalid token.' });
  }
});

export default router;
