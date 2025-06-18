import { RawNodeDatum, CustomNodeElementProps } from "react-d3-tree";

export interface ThoughtNode {
  id: string;
  content: string;
  aspect: string;
  evaluation: string;
  parent: string;
}

export interface ThoughtTreeData {
  nodes: ThoughtNode[];
}

export interface AnalysisResult {
  is_publishable: boolean;
  primary_strengths: string[];
  critical_weaknesses: string[];
  recommendation: string;
  ai_content_percentage: number;
  thought_tree_data: ThoughtTreeData;
}

export interface TreeNodeDatum extends RawNodeDatum {
  name: string;
  attributes: {
    evaluation: string;
    aspect: string;
  };
  children?: TreeNodeDatum[];
}

export interface ZoomState {
  translate: { x: number; y: number };
  scale: number;
}

export interface TreeUpdateState {
  translate: { x: number; y: number };
  zoom: number;
}

export interface CustomNodeProps extends CustomNodeElementProps {
  nodeDatum: TreeNodeDatum;
} 