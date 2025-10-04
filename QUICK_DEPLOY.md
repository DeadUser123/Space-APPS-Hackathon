# 🚀 QUICK DEPLOY - GQ Planets

## Fastest Way to Deploy (3 minutes)

### Using Render.com (RECOMMENDED)

1. **Visit**: https://render.com
2. **Sign up** with GitHub
3. **New Web Service** → Connect `DeadUser123/Space-APPS-Hackathon`
4. **Settings**:
   - Build: `pip install -r requirements.txt`
   - Start: `python app.py`
   - Instance: Free
5. **Click Deploy** ✅

Your app will be live at: `https://gq-planets.onrender.com`

---

## Alternative: Vercel (if you have Pro)

```bash
npm install -g vercel
vercel login
cd /Users/AdvayChandorkar/Desktop/Space-APPS-Hackathon
vercel --prod
```

⚠️ **Warning**: Free tier won't work with PyTorch (too large)

---

## Alternative: Railway

```bash
npm install -g railway
railway login
cd /Users/AdvayChandorkar/Desktop/Space-APPS-Hackathon
railway init
railway up
```

---

## Pre-Deploy Checklist

```bash
# 1. Test locally
python app.py
# Visit http://localhost:5000

# 2. Commit everything
git add .
git commit -m "Ready for deployment"
git push origin main

# 3. Deploy using method above
```

---

## What Gets Deployed

✅ Flask web server  
✅ PyTorch ML model (`checkpoints/best.pth`)  
✅ All 3 pages (Home, Predict, Metrics)  
✅ CSV datasets  
✅ Static files (CSS, JS, images)  

---

## After Deployment

- Test predictions work
- Verify all visualizations load
- Check mobile responsiveness
- Share URL with NASA Space Apps judges! 🎉

---

**Need help?** See full guide: `DEPLOYMENT_GUIDE.md`
