import requests
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def retrieve(query: str, k: int):
    url = "http://0.0.0.0:8666/v1/retrieve"
    payload = {"query": query, "k": k}
    response = requests.post(url, json=payload)
    return response.json()


class RagBasedClassifier:
    def __init__(self):
        self.retrieve = retrieve
        self.client = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.3, max_tokens=1000
        )  # Using LangChain's ChatOpenAI

    def classify(self, paper_content):
        retrieved_content = self.retrieve(paper_content, k=6)

        sys_prompt = """
        You are an expert at classifying papers into one of five conferences. You need to classify into TMLR, NeurIPS, KDD, CVPR, EMNLP.
        Here are some similar chunks present in a vectorstore with papers from these five conferences.
        
        """

        for conference in ["TMLR", "CVPR", "KDD", "EMNLP", "NeurIPS"]:
            current_chunks = ""
            for chunk in retrieved_content:
                if chunk["metadata"]["conference"] == conference:
                    current_chunks += chunk["text"]
                    current_chunks += "\n\n\n"
            if len(current_chunks) > 0:
                sys_prompt += "\n\n\n"
                sys_prompt += conference
                sys_prompt += current_chunks

        user_prompt = """
        Now, classify this paper according to the given guidelines.
        """

        user_prompt += paper_content

        # Call LangChain's ChatOpenAI client with the system and user prompt
        response = self.client.invoke(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # return response["content"].
        return response.content


if __name__ == "__main__":
    mosambi = RagBasedClassifier()
    content = """
        Leveraging Clustering Techniques for Enhanced
        Drone Monitoring and Position Estimation
        Abstract
        Drone tracking and localization are essential for various applications, including
        managing drone formations and implementing anti-drone strategies. Pinpointing
        and monitoring drones in three-dimensional space is difficult, particularly when
        trying to capture the subtle movements of small drones during rapid maneuvers.
        This involves extracting faint signals from varied flight settings and maintaining
        alignment despite swift actions. Typically, cameras and LiDAR systems are used
        to record the paths of drones. However, they encounter challenges in categorizing
        drones and estimating their positions accurately. This report provides an overview
        of an approach named CL-Det. It uses a clustering-based learning detection strategy
        to track and estimate the position of drones using data from two types of LiDAR
        sensors: Livox Avia and LiDAR 360. This method merges data from both LiDAR
        sources to accurately determine the drone’s location in three dimensions. The
        method begins by synchronizing the time codes of the data from the two sensors
        and then isolates the point cloud data for the objects of interest (OOIs) from the
        environmental data. A Density-Based Spatial Clustering of Applications with
        Noise (DBSCAN) method is applied to cluster the OOI point cloud data, and the
        center point of the most prominent cluster is taken as the drone’s location. The
        technique also incorporates past position estimates to compensate for any missing
        information.
        1 Introduction
        Unmanned aerial vehicles (UAVs), commonly referred to as drones, have gained prominence and
        significantly influence areas like logistics, imaging, and emergency response, offering substantial
        advantages to society. However, the broad adoption and sophisticated features of compact, off-the-
        shelf drones have created intricate security issues that extend beyond conventional risks.
        Recent years have witnessed a surge in research on anti-UAV systems. Present anti-UAV methods
        predominantly utilize visual, radar, and radio frequency (RF) technologies. Despite these strides,
        recognizing drones poses a considerable hurdle for sensors like cameras, particularly when drones
        are at significant altitudes or in challenging visual environments. These methods usually fail to spot
        small drones because of their minimal size, which leads to a decreased radar cross-section and a
        less noticeable visual presence. Furthermore, current anti-UAV studies primarily focus on detecting
        objects and tracking them in two dimensions, overlooking the crucial element of estimating their
        3D paths. This omission significantly restricts the effectiveness of anti-UAV systems in practical,
        real-world contexts.
        Our proposed solution, a detection method based on clustering learning (CL-Det), uses the strengths
        of both Livox Avia and LiDAR 360 to improve the tracking and position estimation of UAVs.
        Initially, the timestamps from the Livox Avia and LiDAR 360 data are aligned to maintain temporal
        consistency. By examining the LiDAR data, which contains the spatial coordinates of objects at
        specific times, and comparing these to the actual recorded positions of the drone at those times, the
        drone’s location within the LiDAR point cloud data is effectively pinpointed. The point cloud for
        .
        objects of interest (OOIs) is then isolated from the environmental data. The point cloud of the OOIs
        is grouped using the DBSCAN algorithm, and the central point of the largest cluster is designated as
        the UAV’s position. Moreover, radar data also faces significant challenges due to missing information.
        To mitigate potential data deficiencies, past estimations are employed to supplement missing data,
        thereby maintaining the consistency and precision of UAV tracking.
        2 Methodology
        This section details the methodology employed to ascertain the drone’s spatial position utilizing
        information from LiDAR 360 and Livox Avia sensors. The strategy integrates data from both sensor
        types to achieve precise position calculations.
        2.1 Data Sources
        The following modalities of data were utilized:
        • Double fisheye camera visual images
        • Livox Mid-360 (LiDAR 360) 3D point cloud data
        • Livox Avia 3D point cloud data
        • Millimeter-wave radar 3D point cloud data
        Only 14 out of 59 test sequences have non-zero radar values; therefore, the radar dataset is excluded
        from this work due to data availability issues. Two primary sensor types are employed: LiDAR 360
        and Livox Avia, both of which supply 3D point cloud data crucial for identifying the drone’s location.
        The detailed data descriptions are outlined as follows:
        • LiDAR 360 offers a complete 360-degree view with 3D point cloud data. This dataset
        encompasses environmental details and other observable objects.
        • Livox Avia delivers focused 3D point cloud data at specific timestamps, typically indicating
        the origin point or the drone’s position.
        2.2 Algorithm
        For every sequence, corresponding positions are recorded at specific timestamps. The procedure
        gives precedence to LiDAR 360 data, using Livox Avia data as a backup if the former is not available.
        If neither source is accessible, the position is estimated using historical averages.
        2.2.1 LiDAR 360 Data Processing
        • Separation of Points: The LiDAR 360 data is visually examined to classify areas into two
        zones: environment and non-environment zones.
        • Removal of Environment Points: All points within the environment zone are deemed part
        of the surroundings and are thus excluded from the dataset. After removing environment
        points, it is observed that the remaining non-environment points imply the drone position.
        • Clustering: The DBSCAN clustering algorithm is applied to the remaining points to discern
        distinct clusters.
        • Cluster Selection: The most extensive non-environment cluster is chosen as the representa-
        tive group of points that correspond to the drone.
        • Mean Position Calculation: The drone’s position is determined by calculating the mean of
        the selected cluster, represented by (x, y, z) coordinates.
        2.2.2 Livox Avia Data Processing
        • Removal of Noise: Points with coordinates (0, 0, 0) are eliminated as they are regarded as
        noise.
        • Mean Position Calculation: The mean of the residual points is computed to ascertain the
        drone’s position in (x, y, z) coordinates.
        2
        2.2.3 Fallback Method
        When neither LiDAR 360 nor Livox Avia data is available, the average location of the drone derived
        from training datasets is used. The average ground truth position (x, y, z) from all training datasets
        estimates the drone ground truth position, which is (0.734, -9.739, 33.353).
        2.3 Implementation Details
        The program fetches LiDAR 360 or Livox Avia data from the nearest timestamp for each sequence,
        as indicated in the test dataset. Clustering is executed using the DBSCAN algorithm with appro-
        priate parameters to guarantee strong clustering. Visual inspection is employed for the preliminary
        separation of points, ensuring an accurate categorization of environment points.
        The implementation was conducted on a Lenovo IdeaPad Slim 5 Pro (16") running Windows 11
        with an AMD Ryzen 7 5800H CPU and 16GB DDR4 RAM. The analysis was carried out in a
        Jupyter Notebook environment using Python 3.10. For clustering, the DBSCAN algorithm from the
        Scikit-Learn library was utilized. The DBSCAN algorithm was configured with an epsilon (eps)
        value of 2 and a minimum number of points (minPts) set to 1.
        3 Results
        The algorithm achieved a pose MSE loss of 120.215 and a classification accuracy of 0.322. Table 1
        presents the evaluation results compared to other teams.
        Table 1: Evaluation results on the leaderboard
        Team ID Pose MSE (↓) Accuracy (↑)
        SDUCZS 58198 2.21375 0.8136
        Gaofen Lab 57978 7.299575 0.3220
        sysutlt 57843 24.50694 0.3220
        casetrous 58233 56.880267 0.2542
        NTU-ICG (ours) 58268 120.215107 0.3220
        MTC 58180 189.669428 0.2724
        gzist 56936 417.396317 0.2302
        4 Conclusions
        This paper introduces a clustering-based learning method, CL-Det, which employs advanced cluster-
        ing techniques such as K-Means and DBSCAN for drone detection and position estimation using
        LiDAR data. The approach guarantees dependable and precise drone position estimation by utilizing
        multi-sensor data and robust clustering methods. Fallback mechanisms are in place to ensure con-
        tinuous position estimation even when primary sensor data is absent. Through thorough parameter
        optimization and comparative assessment, the proposed method’s effective performance in drone
        tracking and position estimation is demonstrated.
    """
    print(mosambi.classify(content))
