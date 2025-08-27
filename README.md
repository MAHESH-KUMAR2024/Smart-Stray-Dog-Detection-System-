# Smart-Stray-Dog-Detection-System-
Stray dogs are a common sight in urban areas, often causing safety concerns, road accidents, and challenges for city management. I wanted to create a system that uses AI and computer vision to help monitor and manage stray dog populations in real time, making communities safer and smarter.

Smart Stray Dog Detection System

An AI + Community Web Application platform for detecting, reporting, and managing stray dogs in urban areas.
The system integrates Computer Vision (YOLO-based detection) with a Web Application that enables reporting, donations, and adoptions — bridging the gap between citizens and municipal authorities.

📌 System Architecture
1. Community Engagement Platform (Left side – Purple box)

This is the user-facing part of the system, mainly via a Web Application.
It includes:

📷 Reporting System → Users (or the system) can report stray/detected dogs.

🐾 Adoption Services → Connects rescued dogs with people willing to adopt.

💰 Donation Portal → Allows people to donate money for dog welfare.

👥 User Management → Handles authentication, roles, and permissions for users.

👉 Purpose: Creates a bridge between the community (citizens) and the system.

2. Surveillance and Detection Module (Top Right – Blue box)

This is the core detection system where computer vision works.

🎥 Camera Feed Processor → Takes live video feeds from cameras.

🤖 Detection Engine (Computer Vision Service) → Uses YOLO AI to detect dogs in real time.

✅ Threshold Manager → Ensures that only valid/high-confidence detections are considered.

🔔 Alerting System → If stray dogs are detected, alerts are sent to Municipal Authorities for action.

👉 Purpose: Automatically detect stray dogs from surveillance cameras and alert authorities.

3. Data Storage and Management (Green box)

This is the backend data handling system.

🗄 Database (MongoDB) → Stores detected dog data, reports, adoption records, donations, etc.

📊 Data Analytics → Processes stored data to provide insights (e.g., total stray dogs detected, area-wise distribution, adoption success rate).

👉 Purpose: Acts as the central brain that stores and analyzes all data.

4. Notification and Communication Module (Orange box)

Handles communication with users and authorities.

📧 Email/SMS Notification Service → Sends alerts, adoption updates, donation confirmations, etc.

👉 Purpose: Keeps all stakeholders updated in real time.

5. Municipal Authorities (Right side – Pink box)

This is the external stakeholder who gets alerts whenever a stray dog is detected.

🔗 Connected directly to the Alerting System and Database for reports.

🏥 Helps in actual field action (rescue, vaccination, relocation, etc.).
