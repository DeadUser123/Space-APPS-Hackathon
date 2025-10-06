# Space APPS Hackathon - GQ Planets

### ML Model Training Flowchart

```mermaid
flowchart TD
    KEP["Kepler (KOI) - Space Telescope Data"] --> MERGE["Combine Data and Clean Up"]
    K2["K2 - Space Telescope Data"] --> MERGE
    TESS["TESS (TOI) - Space Telescope Data"] --> MERGE
    MERGE --> FE["Prepare Data (fix gaps, standardize)"]
    FE --> SYN["Add Synthetic Unrealistic Examples (~5-10% of dataset)"]
    SYN --> SPLIT["Split Data for Learning and Testing"]
    SPLIT --> TRAIN["Train the Model (teach computer patterns)"]
    TRAIN --> CKPT["Best Version Saved (epoch 706)"]
    CKPT --> EVAL["Check Performance - Accuracy 83%, ROC-AUC 0.88"]
    EVAL --> ART["Create Outputs (plots and results file)"]
     KEP:::data
     MERGE:::prep
     K2:::data
     TESS:::data
     FE:::prep
     SYN:::prep
     SPLIT:::prep
     TRAIN:::train
     CKPT:::train
     EVAL:::eval
     ART:::art
    classDef data fill:#dbeffc,stroke:#2b6cb0,stroke-width:2px,color:#000,font-size:13px,font-weight:600
    classDef prep fill:#fff7d6,stroke:#b45309,stroke-width:2px,color:#000,font-size:13px,font-weight:600
    classDef train fill:#dff6e9,stroke:#059669,stroke-width:2px,color:#000,font-size:13px,font-weight:600
    classDef eval  fill:#f4e8ff,stroke:#7c3aed,stroke-width:2px,color:#000,font-size:13px,font-weight:600
    classDef art   fill:#fff3e0,stroke:#b7791f,stroke-width:2px,color:#000,font-size:13px,font-weight:600
```

### Web

```mermaid
flowchart TD
 subgraph API["Available Endpoints"]
        PRED["Single Prediction (JSON)"]
        BATCH["Batch Prediction (CSV upload)"]
        VIS["Visualizations"]
  end
    CKPT["Best Model Saved (epoch 706)"] --> LOAD["Load Model"]
    LOAD --> START["Start Server"]
    START --> API
    STATIC["Static Visuals (images and charts)"] --> VIS
    PRED --> REQ1["User sends data (JSON)"]
    REQ1 --> PRE1["Prepare input"]
    PRE1 --> MODEL1["Run model → prediction"]
    MODEL1 --> POST1["Format result"]
    POST1 --> RESP1["Send back result (JSON)"]
    BATCH --> FILE["Upload CSV file"]
    FILE --> PARSE["Read CSV rows"]
    PARSE --> LOOP["Process each row with model"]
    LOOP --> AGG["Combine results into table"]
    AGG --> RESP2["Send back results (JSON)"]
    API --> PRED & BATCH & VIS
     PRED:::endpoint
     BATCH:::endpoint
     VIS:::endpoint
     CKPT:::artifact
     START:::server
     STATIC:::artifact
     REQ1:::flow
     PRE1:::flow
     MODEL1:::flow
     POST1:::flow
     RESP1:::flow
     FILE:::flow
     PARSE:::flow
     LOOP:::flow
     AGG:::flow
     RESP2:::flow
    classDef artifact fill:#fff3e0,stroke:#b45309,stroke-width:2px,color:#000,font-size:13px,font-weight:600
    classDef server fill:#e6fffb,stroke:#059669,stroke-width:2px,color:#000,font-size:13px,font-weight:600
    classDef endpoint fill:#e8f6ff,stroke:#2563eb,stroke-width:2px,color:#000,font-size:13px,font-weight:600
    classDef flow fill:#fff9db,stroke:#b7791f,stroke-width:1.5px,color:#000,font-size:12px,font-weight:600
```
