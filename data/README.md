# IoT Lab Capture
- Description: 
- Probable Malware Name: Mirai
- MD5: 82062b666f09fc5c0fe4f68d1ea90916 
- SHA1: b35d4b7980d361874e84e76eddbaff83c2c5790f 
- SHA256: 49fd1cb22e0325c1f9038160da534fc23672e5509e903a94ce5bcddc893eb2c0 
- Password of zip file: infected
- Duration: 86397.071327 seconds 

- [VirusTotal](https://www.virustotal.com/en/file/49fd1cb22e0325c1f9038160da534fc23672e5509e903a94ce5bcddc893eb2c0/analysis/)
- [HybridAnalysis](https://www.hybrid-analysis.com/sample/49fd1cb22e0325c1f9038160da534fc23672e5509e903a94ce5bcddc893eb2c0?environmentId=2)
- RobotHash

[![](https://robohash.org/82062b666f09fc5c0fe4f68d1ea90916)](https://robohash.org)

# Description of Files

- .capinfos
    - Capinfos file
- .dnstop
    - DNS top file
- .passivedns
    - Passive DNS file
- .pcap
    - Original pcap file
- bro
    - Folder with all the bro output files

# IP Addresses
    - Infected device: 192.168.1.195
    - Default GW: 192.168.1.1

# Generic Dataset name: CTU-IoT-Malware-Capture-34
# Origin device: RPi02

# Timeline

## Start. 2018/12/21
## Fri Dec 21 15:49:28 CET 2018

# Disclaimer 
These files were generated in the Stratosphere Laboratory as part of the Aposemat Project for collecting IoT malware captures Done in the CVUT University, Prague, Czech Republic.
The goal is to store long-lived real iot malware traffic and to generate labeled netflows files.
Any question feel free to contact us at:
Sebastian Garcia: sebastian.garcia@agents.fel.cvut.cz

You need authorization from the Stratosphere Lab to use these files.

# Suricata run with rules updated on 2022-07-16

## Datasets added 2026-11-11 (downloaded with kagglehub)

| Folder | Paper role | Kaggle link | Notes |
|---|---|---|---|
| `microsoft_guide/` | Categorical-dense (replaces Kaggle Cyber) | https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction | Microsoft GUIDE: real security alerts from 6,100+ orgs. 9.5M train rows, label `IncidentGrade` (TruePositive / BenignPositive / FalsePositive, ~2:1.6:1). Nearly all features are categorical IDs. Real-data RF macro-F1 ≈ 0.81 (400k-row sample). Paper: Freitas et al., "AI-Driven Guided Response for Security Operation Centers with Microsoft Copilot for Security", 2024. |
| `sdn_anomaly/` | Explicit graph (SDN Anomaly) | https://www.kaggle.com/datasets/aikenkazin/ddos-sdn-dataset | DDoS attack SDN dataset (Ahuja et al., Mendeley Data, 2020). 104,345 rows, 22 features + `label` (0/1, ~61:39). `src`/`dst` IPs give the graph. Very easy to separate: `bytecount` alone gives F1 ≈ 0.996. |
| `unsw_nb15/` | Backup (not currently used) | https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 | Real traffic, official train/test split. Only 3 categorical features (proto, service, state), so it is NOT categorical-dense. Real-data RF macro-F1 ≈ 0.89. |
| `kaggle_cyber/` | Original "Kaggle Cyber" (do not use) | https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks | Randomly generated: the label is independent of every feature, so real-data F1 ≈ 0.336 (chance). |
