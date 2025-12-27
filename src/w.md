graph TD
    %% Styling
    classDef storage fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef hardware fill:#4caf50,stroke:#1b5e20,stroke-width:2px,color:white;

    %% --- MAIN THREAD ---
    subgraph Main_Thread [Main Thread: Setup & UI]
        Start([Start Program]) --> ParseArgs[Parse CLI Arguments]
        ParseArgs --> CheckMode{Passthrough?}
        
        CheckMode -- No --> LoadModel[Load PyTorch Model\nFMModelLoader]
        CheckMode -- Yes --> SetPass[Set Model = None]
        
        LoadModel --> InitClass[Init LiveSDRDenoiser]
        SetPass --> InitClass
        
        InitClass --> FindDev[Find Virtual Cable & Speaker IDs]
        FindDev --> StartAI[Start AI Thread]
        StartAI --> StartStream[Start sounddevice Stream]
        
        StartStream --> PlotCheck{Plot Enabled?}
        PlotCheck -- Yes --> UpdatePlot[Update Matplotlib\nWaveform & Spectrum]
        UpdatePlot --> Sleep[Sleep 0.1s]
        Sleep --> PlotCheck
        PlotCheck -- No --> Wait[Wait for KeyboardInterrupt]
    end

    %% --- DATA QUEUES ---
    InQ[(Input Queue)]:::storage
    OutQ[(Output Queue)]:::storage

    %% --- AUDIO CALLBACK THREAD ---
    subgraph Audio_Thread [Audio Callback Thread]
        direction TB
        CableInput((Virtual Cable Input)):::hardware --> ReadAudio[Read Raw Audio Chunk]
        ReadAudio --> |Push| InQ
        
        CheckOutQ{Out Queue Empty?}
        CheckOutQ -- No --> GetAudio[Get Denoised Chunk]
        CheckOutQ -- Yes --> Fallback[Use Last Output Chunk\nAvoid Glitches]
        
        GetAudio --> WriteAudio[Write to Output Buffer]
        Fallback --> WriteAudio
        WriteAudio --> Speakers((Speakers)):::hardware
    end

    %% --- AI WORKER THREAD ---
    subgraph AI_Thread [AI Worker Thread]
        direction TB
        WaitInQ[Wait for Input Queue] --> |Pop Chunk| Process{Passthrough Mode?}
        InQ --> WaitInQ
        
        Process -- Yes --> HistPass[Update History]
        HistPass --> |Push Raw| OutQ
        
        Process -- No --> PreProcess[To Tensor \n unsqueeze dimensions]
        PreProcess --> Inference[Model Inference\n U-Net / STFT]
        Inference --> PostProcess[To Numpy \n squeeze dimensions]
        PostProcess --> HistAI[Update History]
        HistAI --> |Push Clean| OutQ
    end

    %% Connections between subgraphs
    StartStream -.-> |Spawns| Audio_Thread
    StartAI -.-> |Spawns| AI_Thread
    
    %% Plotting connections
    HistAI -.-> |Read Deque| UpdatePlot
    HistPass -.-> |Read Deque| UpdatePlot