# 🚀 GQ Planets - Complete Deployment Guide

## ⚠️ IMPORTANT: Vercel Limitations with PyTorch

**Vercel is NOT recommended for this project** because:
- PyTorch is ~700MB (Vercel limit: 50MB per function)
- Serverless architecture causes cold starts (slow first load)
- Better alternatives exist for ML applications

## ✅ RECOMMENDED: Deploy to Render.com (Best for ML Apps)

### Why Render?
- ✅ **FREE tier** perfect for demos
- ✅ **Supports PyTorch** and large ML models
- ✅ **No cold starts** - persistent server
- ✅ **Easy deployment** from GitHub
- ✅ **Auto-deploy** on git push

### Step-by-Step Render Deployment

#### 1. Prepare Your Repository

```bash
# Make sure all files are committed
git add .
git commit -m "Add deployment configuration"
git push origin main
```

#### 2. Deploy to Render

1. **Go to**: https://render.com
2. **Sign up** with your GitHub account
3. Click **"New +"** → **"Web Service"**
4. **Connect** your repository: `DeadUser123/Space-APPS-Hackathon`

#### 3. Configure the Service

Fill in these settings:

- **Name**: `gq-planets` (or any name you like)
- **Region**: Choose closest to you
- **Branch**: `main`
- **Root Directory**: Leave empty
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`
- **Instance Type**: `Free`

#### 4. Environment Variables (Optional)

No environment variables needed for basic deployment.

#### 5. Click "Create Web Service"

Render will:
1. Clone your repository
2. Install dependencies from `requirements.txt`
3. Run `python app.py`
4. Give you a live URL like: `https://gq-planets.onrender.com`

⏱️ **First deployment takes ~5-10 minutes** (installing PyTorch)

---

## Alternative Option: Railway.app

### Why Railway?
- ✅ Free $5/month credit (enough for demo)
- ✅ Very fast deployments
- ✅ Great for Python/ML apps
- ✅ Simple CLI deployment

### Railway Deployment Steps

#### 1. Install Railway CLI

```bash
npm install -g railway
```

#### 2. Login and Deploy

```bash
# Login to Railway
railway login

# Initialize project (in your project directory)
cd /Users/AdvayChandorkar/Desktop/Space-APPS-Hackathon
railway init

# Deploy
railway up

# Open in browser
railway open
```

#### 3. Set Start Command

In Railway dashboard:
- Go to your service
- Settings → Deploy
- Start Command: `python app.py`
- Save changes

---

## 🔧 Optimization Before Deployment

### 1. Clean Up Checkpoint Files (Optional)

You have 100+ checkpoint files. Keep only `best.pth`:

```bash
./cleanup_checkpoints.sh
```

Or manually:
```bash
cd checkpoints
rm checkpoint_epoch_*.pth
# Keep only best.pth
```

### 2. Verify Requirements

Make sure `requirements.txt` has all dependencies:
```
torch>=2.0.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
flask>=2.3.0
plotly>=5.14.0
seaborn>=0.12.0
```

### 3. Test Locally First

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Visit http://localhost:5000
# Test all features before deploying
```

---

## 📱 If You MUST Use Vercel

### Option 1: Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
cd /Users/AdvayChandorkar/Desktop/Space-APPS-Hackathon
vercel

# Deploy to production
vercel --prod
```

### Option 2: Vercel GitHub Integration

1. Go to https://vercel.com/dashboard
2. Click "Add New Project"
3. Import `DeadUser123/Space-APPS-Hackathon`
4. Click "Deploy"

### ⚠️ Expected Vercel Issues

**Problem**: "Serverless Function size limit exceeded"
**Cause**: PyTorch is too large (~700MB)

**Solutions**:
1. **Upgrade to Vercel Pro** ($20/month) for larger limits
2. **Convert to ONNX** (lighter runtime, more complex)
3. **Use Render/Railway instead** (recommended)

---

## 🎯 Quick Start Guide (TL;DR)

### For NASA Space Apps Demo - Use Render

1. **Commit your code**:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Go to Render.com**:
   - Sign up with GitHub
   - New Web Service
   - Connect repository
   - Use these settings:
     - Build: `pip install -r requirements.txt`
     - Start: `python app.py`
     - Free tier

3. **Wait 5-10 minutes** for first deployment

4. **Get your URL**: `https://gq-planets.onrender.com`

5. **Share with judges** 🎉

---

## 📋 Deployment Checklist

Before deploying, make sure:

- [ ] All code is committed and pushed to GitHub
- [ ] `requirements.txt` includes all dependencies
- [ ] App runs successfully locally (`python app.py`)
- [ ] `checkpoints/best.pth` exists
- [ ] All three pages work (/, /predict, /metrics)
- [ ] Predictions show realistic confidence (80-87%)
- [ ] All visualizations display correctly

---

## 🐛 Troubleshooting

### "Application failed to respond"
- Check start command is `python app.py`
- Ensure port is set to `5000` or use `PORT` environment variable
- Check build logs for errors

### "Module not found"
- Verify dependency is in `requirements.txt`
- Check spelling and version compatibility

### "Out of memory"
- PyTorch uses significant memory
- Upgrade to paid tier or use lighter model

### Slow first load
- Normal for serverless (Vercel)
- Use Render/Railway for persistent servers
- Model loading takes 2-3 seconds first time

---

## 📊 Platform Comparison

| Platform | Free Tier | ML Support | Cold Starts | Deploy Time | Best For |
|----------|-----------|------------|-------------|-------------|----------|
| **Render** | ✅ Yes | ✅ Excellent | ❌ No | ~10 min | **ML Demos** |
| **Railway** | ✅ $5 credit | ✅ Excellent | ❌ No | ~5 min | **Quick Deploy** |
| Vercel | ✅ Yes | ⚠️ Limited | ✅ Yes | ~3 min | Web Apps |
| Heroku | ❌ Paid only | ✅ Good | ❌ No | ~8 min | Legacy Apps |

**Winner for GQ Planets**: **Render.com** 🏆

---

## 🎬 Next Steps

1. Choose your platform (Render recommended)
2. Follow the deployment steps above
3. Test your live URL
4. Add the URL to your NASA Space Apps submission
5. Celebrate! 🎉

---

## 📞 Need Help?

Common issues:
- Deployment fails → Check build logs in platform dashboard
- App crashes → Verify `python app.py` works locally
- Wrong page shows → Check routes in `app.py`
- 502/503 errors → Wait a few minutes, service might be starting

## Files Created for Deployment

✅ `vercel.json` - Vercel configuration
✅ `wsgi.py` - WSGI entry point  
✅ `.vercelignore` - Files to exclude from Vercel
✅ `Procfile` - For Heroku-compatible platforms
✅ `runtime.txt` - Python version specification
✅ `cleanup_checkpoints.sh` - Cleanup script

All platforms should work with these files!
