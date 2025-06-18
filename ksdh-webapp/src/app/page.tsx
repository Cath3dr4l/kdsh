"use client";

import { useState, useCallback } from "react";
import { Button } from "~/components/ui/button";
import { Textarea } from "~/components/ui/textarea";
import { Input } from "~/components/ui/input";
import Tree from "react-d3-tree";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "~/components/ui/card";
import { MinusIcon, PlusIcon, TimerResetIcon } from "lucide-react";
import { CustomNode } from "~/components/CustomNode";
import { transformToTreeData } from "~/utils/treeTransform";
import { AnalysisResult, ZoomState, TreeUpdateState } from "~/types";

// Update the API base URL
const API_BASE_URL = "http://0.0.0.0:8000";

// Add this type definition at the top with other types
type ClassificationResult = {
  final_prediction: {
    conference: string;
    rationale: string;
  };
  llm_prediction: {
    conference: string;
    rationale: string;
    thought_process: string[];
  };
  rag_prediction: {
    conference: string;
    rationale: string;
  };
  similarity_prediction: {
    conference: string;
    rationale: string;
  };
};

export default function HomePage() {
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [zoomState, setZoomState] = useState<ZoomState>({
    translate: { x: 350, y: 50 },
    scale: 1,
  });
  const [classificationResult, setClassificationResult] =
    useState<ClassificationResult | null>(null);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (file && file.type === "application/pdf") {
      setResult(null);
      setClassificationResult(null);
      setIsLoading(true);

      // Check cache first
      const cacheKey = `pdf-analysis-${file.name}`;
      const cachedData = localStorage.getItem(cacheKey);

      if (cachedData) {
        // If cached data exists, set a timeout to display it
        setTimeout(() => {
          const { evaluateData, classifyData } = JSON.parse(cachedData);
          setResult(evaluateData);
          if (evaluateData.is_publishable) {
            setClassificationResult(classifyData);
          }
          setIsLoading(false);
        }, 20000); // 20 second delay
        return;
      }

      try {
        const formData1 = new FormData();
        const formData2 = new FormData();
        formData1.append("file", file);
        formData2.append("file", file);

        // Make both requests in parallel
        const [evaluateResponse, classifyResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/evaluate/pdf`, {
            method: "POST",
            body: formData1,
          }),
          fetch(`${API_BASE_URL}/classify/pdf`, {
            method: "POST",
            body: formData2,
          }),
        ]);

        if (!evaluateResponse.ok) {
          throw new Error("PDF analysis failed");
        }

        const evaluateData = await evaluateResponse.json();
        let classifyData = null;

        // Only process classification result if document is publishable
        if (evaluateData.is_publishable && classifyResponse.ok) {
          classifyData = await classifyResponse.json();
          setClassificationResult(classifyData);
        }

        // Cache the results
        localStorage.setItem(
          cacheKey,
          JSON.stringify({
            evaluateData,
            classifyData,
            timestamp: new Date().getTime(),
          }),
        );

        setResult(evaluateData);
      } catch (error) {
        console.error("Error:", error);
      } finally {
        setIsLoading(false);
      }
    } else {
      alert("Please upload a PDF file");
    }
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    try {
      setResult(null);
      setClassificationResult(null);
      // Make both requests in parallel
      const [evaluateResponse, classifyResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/evaluate/text`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ content: inputText }),
        }),
        fetch(`${API_BASE_URL}/classify/text`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ content: inputText }),
        }),
      ]);

      if (!evaluateResponse.ok) {
        throw new Error("Analysis failed");
      }

      const evaluateData = await evaluateResponse.json();
      setResult(evaluateData);

      // Only set classification result if document is publishable
      if (evaluateData.is_publishable && classifyResponse.ok) {
        const classifyData = await classifyResponse.json();
        setClassificationResult(classifyData);
      }
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Update the zoom state handler to handle mouse wheel and drag events
  const handleTreeUpdate = useCallback((state: TreeUpdateState) => {
    setZoomState((prev) => {
      // Only update if values have changed to prevent unnecessary rerenders
      if (
        prev.scale === state.zoom &&
        prev.translate.x === state.translate.x &&
        prev.translate.y === state.translate.y
      ) {
        return prev;
      }

      // Limit translation to prevent dragging too far
      const h = 600; // height of tree container
      const w = 1500; // width of tree container
      const scale = state.zoom;
      const tbound = -h * scale;
      const bbound = h * scale;
      const lbound = -w * scale;
      const rbound = w * scale;

      // Constrain translation within bounds
      const translation = {
        x: Math.max(Math.min(state.translate.x, rbound), lbound),
        y: Math.max(Math.min(state.translate.y, bbound), tbound),
      };

      return {
        scale: scale,
        translate: translation,
      };
    });
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center bg-gradient-to-b from-slate-950 to-slate-900 p-8 text-slate-50">
      <div className="container max-w-5xl">
        <div className="mb-12 text-center">
          <h1 className="mb-4 bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-5xl font-extrabold tracking-tight text-transparent">
            Research Paper Analysis Assistant
          </h1>
          <p className="text-lg text-slate-400">
            Unlock powerful insights with our AI-powered research paper analyzer
          </p>
        </div>

        <Card className="mb-8 border-slate-800 bg-slate-900/50 shadow-lg backdrop-blur">
          <CardHeader>
            <CardTitle className="text-slate-200">Input Methods</CardTitle>
            <CardDescription className="text-slate-400">
              Choose to either upload a PDF document or enter text directly
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-200">
                Upload PDF
              </label>
              <Input
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                className="cursor-pointer border-slate-700 bg-slate-800 text-slate-200 hover:bg-slate-800/80"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-200">
                Or enter text
              </label>
              <Textarea
                placeholder="Type or paste your text here..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="min-h-[200px] border-slate-700 bg-slate-800 text-slate-200 placeholder:text-slate-500"
              />
            </div>

            <Button
              onClick={handleSubmit}
              disabled={isLoading || (!inputText && !File)}
              className="w-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 text-white transition-all hover:from-blue-600 hover:via-purple-600 hover:to-pink-600"
            >
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
                  Processing...
                </div>
              ) : (
                "Analyze"
              )}
            </Button>
          </CardContent>
        </Card>

        {result && (
          <div className="container max-w-5xl space-y-8">
            <Card className="border-slate-800 bg-slate-900/50 shadow-lg backdrop-blur">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-slate-200">
                  Analysis Result
                  <span
                    className={`rounded-full px-3 py-1 text-sm ${
                      result.is_publishable
                        ? "bg-green-500/20 text-green-400"
                        : "bg-red-500/20 text-red-400"
                    }`}
                  >
                    {result.is_publishable ? "Publishable" : "Not Publishable"}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* AI Content Percentage */}
                <div>
                  <h3 className="mb-2 font-medium text-slate-200">
                    AI Content Percentage
                  </h3>
                  <div className="h-3 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
                      style={{ width: `${result.ai_content_percentage}%` }}
                    ></div>
                  </div>
                  <p className="mt-2 text-sm text-slate-400">
                    {result.ai_content_percentage.toFixed(1)}% AI-generated
                    content
                  </p>
                </div>

                {/* Strengths */}
                {result.primary_strengths.length > 0 && (
                  <div className="rounded-lg bg-green-500/10 p-4">
                    <h3 className="mb-2 font-medium text-green-400">
                      Primary Strengths
                    </h3>
                    <ul className="list-inside list-disc space-y-2">
                      {result.primary_strengths.map((strength, index) => (
                        <li key={index} className="text-slate-300">
                          {strength}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Weaknesses */}
                {result.critical_weaknesses.length > 0 && (
                  <div className="rounded-lg bg-red-500/10 p-4">
                    <h3 className="mb-2 font-medium text-red-400">
                      Critical Weaknesses
                    </h3>
                    <ul className="list-inside list-disc space-y-2">
                      {result.critical_weaknesses.map((weakness, index) => (
                        <li key={index} className="text-slate-300">
                          {weakness}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Recommendation */}
                <div className="rounded-lg bg-slate-800/50 p-4">
                  <h3 className="mb-2 font-medium text-slate-200">
                    Recommendation
                  </h3>
                  <p className="text-slate-400">{result.recommendation}</p>
                </div>
              </CardContent>
            </Card>

            {/* Conference Classification Card */}
            {result && result.is_publishable && classificationResult && (
              <Card className="mb-8 border-slate-800 bg-slate-900/50 shadow-lg backdrop-blur">
                <CardHeader>
                  <CardTitle className="text-slate-200">
                    Conference Classification
                  </CardTitle>
                  <CardDescription className="text-slate-400">
                    Analysis of the most suitable conference for publication
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Final Prediction */}
                  <div className="rounded-lg bg-gradient-to-r from-green-500/10 to-emerald-500/10 p-6">
                    <h3 className="mb-2 font-semibold text-green-400">
                      Recommended Conference:{" "}
                      <span className="text-emerald-300">
                        {classificationResult.final_prediction.conference}
                      </span>
                    </h3>
                    <p className="text-slate-300">
                      {classificationResult.final_prediction.rationale}
                    </p>
                  </div>

                  {/* Analysis Sections - Updated to include similarity analysis */}
                  <div className="grid gap-4 md:grid-cols-3">
                    {/* LLM Analysis */}
                    <div className="rounded-lg border border-slate-800 bg-slate-800/30 p-4">
                      <h4 className="mb-2 font-medium text-blue-400">
                        Language Model Analysis
                      </h4>
                      <p className="text-sm text-slate-400">
                        {classificationResult.llm_prediction.rationale}
                      </p>
                    </div>

                    {/* RAG Analysis */}
                    <div className="rounded-lg border border-slate-800 bg-slate-800/30 p-4">
                      <h4 className="mb-2 font-medium text-purple-400">
                        RAG - Based Analysis
                      </h4>
                      <p className="text-sm text-slate-400">
                        {classificationResult.rag_prediction.rationale}
                      </p>
                    </div>

                    {/* Similarity Analysis */}
                    <div className="rounded-lg border border-slate-800 bg-slate-800/30 p-4">
                      <h4 className="mb-2 font-medium text-amber-400">
                        Similarity Analysis
                      </h4>
                      <div className="space-y-2 text-sm text-slate-400">
                        {classificationResult.similarity_prediction.rationale}
                      </div>
                    </div>
                  </div>

                  {/* Thought Process */}
                  {classificationResult.llm_prediction.thought_process && (
                    <div className="rounded-lg border border-slate-800 bg-slate-800/30 p-4">
                      <h5 className="mb-3 text-sm font-medium text-pink-400">
                        Analysis Breakdown:
                      </h5>
                      <ul className="space-y-2 text-sm text-slate-400">
                        {classificationResult.llm_prediction.thought_process.map(
                          (thought, index) => (
                            <li key={index} className="flex items-start gap-2">
                              <span className="mt-1 h-2 w-2 rounded-full bg-pink-500"></span>
                              {thought}
                            </li>
                          ),
                        )}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Thought Tree Card */}
            <Card className="overflow-hidden">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span>Thought Process Tree</span>
                  <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-500"></div>
                </CardTitle>
                <CardDescription>
                  Visual representation of the analysis thought process
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <div className="relative h-[600px] w-full overflow-hidden bg-gradient-to-b from-background to-background/50">
                  <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,0,0,0.02),transparent)] dark:bg-[radial-gradient(circle_at_50%_50%,rgba(255,255,255,0.02),transparent)]" />

                  {/* Zoom controls */}
                  <div className="absolute right-4 top-4 z-10 flex gap-2">
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => {
                        setZoomState((prev) => ({
                          ...prev,
                          scale: prev.scale * 1.2,
                        }));
                      }}
                    >
                      <PlusIcon className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => {
                        setZoomState((prev) => ({
                          ...prev,
                          scale: prev.scale / 1.2,
                        }));
                      }}
                    >
                      <MinusIcon className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => {
                        setZoomState({
                          translate: { x: 350, y: 50 },
                          scale: 1,
                        });
                      }}
                    >
                      <TimerResetIcon className="h-4 w-4" />
                    </Button>
                  </div>

                  {/* Add a container div with specific dimensions */}
                  <div
                    className="h-full w-full"
                    style={{ minWidth: "800px", minHeight: "800px" }}
                  >
                    <Tree
                      data={transformToTreeData(result.thought_tree_data.nodes)}
                      orientation="vertical"
                      pathFunc="step"
                      translate={zoomState.translate}
                      zoom={zoomState.scale}
                      nodeSize={{ x: 170, y: 150 }}
                      separation={{ siblings: 1.5, nonSiblings: 1.75 }}
                      renderCustomNodeElement={(rd3tProps) => (
                        <CustomNode {...rd3tProps} />
                      )}
                      pathClassFunc={() =>
                        "stroke-gray-300 dark:stroke-gray-600 stroke-2 transition-all duration-300"
                      }
                      onUpdate={handleTreeUpdate}
                      // enableLegacyTransitions={true}
                      transitionDuration={100}
                      zoomable={true}
                      draggable={true}
                      scaleExtent={{ min: 0.1, max: 3 }}
                      initialDepth={3}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </main>
  );
}
