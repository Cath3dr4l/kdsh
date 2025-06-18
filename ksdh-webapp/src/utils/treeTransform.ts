import { ThoughtNode, TreeNodeDatum } from "~/types";

export function transformToTreeData(nodes: ThoughtNode[]): TreeNodeDatum {
  console.log(nodes);
  if (!nodes.length) {
    return {
      name: "No data available",
      attributes: {
        evaluation: "neutral",
        aspect: "none",
      },
      children: [],
    };
  }

  const nodeMap = new Map<string, TreeNodeDatum>();
  let root: TreeNodeDatum | null = null;

  // Create node objects
  nodes.forEach((node) => {
    nodeMap.set(node.id, {
      name: node.content.substring(0, 50) + "...",
      attributes: {
        evaluation: node.evaluation,
        aspect: node.aspect,
      },
      children: [],
    });
  });

  // Build tree structure
  nodes.forEach((node) => {
    const treeNode = nodeMap.get(node.id);
    if (!treeNode) return;

    if (node.parent) {
      const parentNode = nodeMap.get(node.parent);
      if (parentNode && parentNode.children) {
        parentNode.children.push(treeNode);
      }
    } else {
      root = treeNode;
    }
  });

  return root || nodeMap.values().next().value;
}
