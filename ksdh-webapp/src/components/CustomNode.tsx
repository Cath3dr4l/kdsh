import { motion } from "framer-motion";
import { CustomNodeProps } from "~/types";

export const CustomNode: React.FC<CustomNodeProps> = ({ nodeDatum, toggleNode }) => (
  <motion.g
    initial={{ opacity: 0, scale: 0.8 }}
    animate={{ opacity: 1, scale: 1 }}
    transition={{ duration: 0.3 }}
  >
    {/* Glow effect */}
    <circle
      r={22}
      className={`opacity-20 blur-sm ${
        nodeDatum.attributes.evaluation === "best"
          ? "fill-green-400"
          : nodeDatum.attributes.evaluation === "neutral"
            ? "fill-yellow-400"
            : "fill-red-400"
      }`}
    />

    {/* Main circle with gradient */}
    <circle
      r={20}
      onClick={toggleNode}
      className={`cursor-pointer transition-all duration-300 hover:brightness-110 ${
        nodeDatum.attributes.evaluation === "best"
          ? "fill-[url(#greenGradient)]"
          : nodeDatum.attributes.evaluation === "neutral"
            ? "fill-[url(#yellowGradient)]"
            : "fill-[url(#redGradient)]"
      }`}
    />

    {/* Evaluation badge */}
    <foreignObject x={-25} y={-35} width={50} height={20}>
      <div
        className={`rounded-full px-2 py-0.5 text-center text-[10px] font-medium shadow-sm ${
          nodeDatum.attributes.evaluation === "best"
            ? "bg-green-100 text-green-800"
            : nodeDatum.attributes.evaluation === "neutral"
              ? "bg-yellow-100 text-yellow-800"
              : "bg-red-100 text-red-800"
        }`}
      >
        {nodeDatum.attributes.evaluation}
      </div>
    </foreignObject>

    {/* Content card */}
    <foreignObject x={30} y="-30" width="220" height="120">
      <motion.div
        initial={{ x: -10, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.3, delay: 0.1 }}
        className="rounded-lg border bg-gradient-to-b from-white/95 to-white/90 p-3 shadow-lg backdrop-blur-sm dark:from-gray-800/95 dark:to-gray-800/90"
      >
        <div className="text-xs font-medium text-gray-700 dark:text-gray-200">
          {nodeDatum.name}
        </div>
        <div className="mt-1 text-[10px] text-gray-500 dark:text-gray-400">
          {nodeDatum.attributes.aspect}
        </div>
      </motion.div>
    </foreignObject>

    {/* Gradient definitions */}
    <defs>
      <radialGradient id="greenGradient" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor="#4ade80" />
        <stop offset="100%" stopColor="#22c55e" />
      </radialGradient>
      <radialGradient id="yellowGradient" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor="#fde047" />
        <stop offset="100%" stopColor="#facc15" />
      </radialGradient>
      <radialGradient id="redGradient" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor="#f87171" />
        <stop offset="100%" stopColor="#ef4444" />
      </radialGradient>
    </defs>
  </motion.g>
); 