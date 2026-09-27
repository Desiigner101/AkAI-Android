package com.akai.service

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import com.google.mediapipe.framework.image.BitmapImageBuilder
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.ArrayDeque

class FSLRecognitionService(private val context: Context) {

    enum class RecognitionMode { WORDS, LETTERS }

    private val TAG = "FSLRecognitionService"

    // Model config
    private val MODEL_FILE        = "akai_model.tflite"
    private val ACTIONS_FILE      = "actions.txt"
    private val SEQ_LENGTH_FILE   = "sequence_length.txt"
    private val FEATURE_DIM       = 138
    private val CONFIDENCE_THRESHOLD = 0.35f
    private val STABLE_REQUIRED   = 2
    private val PREDICT_EVERY_N   = 3
    private val MIN_FRAMES_PCT    = 0.15f
    private val HAND_UPDATE_EVERY = 2  // Run HandLandmarker every 2 frames
    private val POSE_UPDATE_EVERY = 5  // Run PoseLandmarker every 5 frames

    private var interpreter: Interpreter? = null
    private var actions: List<String> = emptyList()
    private var sequenceLength: Int = 120

    // Buffers
    private val frameBuffer    = ArrayDeque<FloatArray>()
    private val predHistory    = ArrayDeque<String>()
    private var frameCount     = 0
    private var lastOutput     = ""
    private var lastOutputTime = 0L
    @Volatile
    var recognitionMode = RecognitionMode.WORDS
    private val COOLDOWN_MS    = 800L
    private var noHandCount    = 0
    private val NO_HAND_TOLERANCE = 5  // Allow 5 frames of no hand before clearing

    // MediaPipe — Hand + Pose (matches Python Holistic)
    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null
    private var cachedPoseResult: PoseLandmarkerResult? = null
    private var cachedHandResult: HandLandmarkerResult? = null
    private var poseFrameCounter = 0
    private var handFrameCounter = 0

    // Callbacks
    var onGestureRecognized: ((String) -> Unit)? = null  // word signs → direct to thread
    var onLetterDetected: ((String) -> Unit)? = null     // letters → go to buffer
    var onTop3Ready: ((List<Pair<String, Float>>) -> Unit)? = null  // top 3 predictions

    fun loadModel() {
        try {
            // Load sequence length
            sequenceLength = context.assets.open(SEQ_LENGTH_FILE)
                .bufferedReader().readText().trim().toInt()
            Log.d(TAG, "Sequence length: $sequenceLength")

            // Load actions
            actions = context.assets.open(ACTIONS_FILE)
                .bufferedReader().readLines().filter { it.isNotBlank() }
            Log.d(TAG, "Actions loaded: ${actions.size} gestures — $actions")

            // Load TFLite model
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options().apply { setNumThreads(2) }
            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "TFLite model loaded")

            // Init HandLandmarker
            val handBaseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()
            val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(handBaseOptions)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setRunningMode(RunningMode.IMAGE)
                .build()
            handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
            Log.d(TAG, "HandLandmarker ready")

            // Init PoseLandmarker — gives us nose + shoulders + elbow
            val poseBaseOptions = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker.task")
                .build()
            val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(poseBaseOptions)
                .setMinPoseDetectionConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setRunningMode(RunningMode.IMAGE)
                .build()
            poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)
            Log.d(TAG, "PoseLandmarker ready")

        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "MediaPipe not available on this device/emulator: ${e.message}")
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed: ${e.message}")
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    fun processFrame(imageProxy: ImageProxy) {
        try {
            val rawBitmap = imageProxy.toBitmap()
            val bitmap = if (rawBitmap.config != Bitmap.Config.ARGB_8888) {
                rawBitmap.copy(Bitmap.Config.ARGB_8888, false)
            } else rawBitmap

            val rotation = imageProxy.imageInfo.rotationDegrees.toFloat()
            val rotated  = rotateBitmap(bitmap, rotation)
            val mpImage  = BitmapImageBuilder(rotated).build()

            // HandLandmarker every 2 frames
            handFrameCounter++
            if (handFrameCounter % HAND_UPDATE_EVERY == 0) {
                cachedHandResult = handLandmarker?.detect(mpImage)
            }

            // PoseLandmarker every 5 frames
            poseFrameCounter++
            if (poseFrameCounter % POSE_UPDATE_EVERY == 0) {
                cachedPoseResult = poseLandmarker?.detect(mpImage)
            }

            val keypoints = extractKeypoints(cachedHandResult, cachedPoseResult)

            if (keypoints != null) {
                noHandCount = 0
                frameBuffer.addLast(keypoints)
                if (frameBuffer.size > sequenceLength) frameBuffer.removeFirst()
                frameCount++

                val minFrames = (sequenceLength * MIN_FRAMES_PCT).toInt()
                if (frameCount % PREDICT_EVERY_N == 0 && frameBuffer.size >= minFrames) {
                    runInference()
                }
            } else {
                // Tolerance — don't clear immediately on brief hand loss (helps Z motion)
                noHandCount++
                if (noHandCount >= NO_HAND_TOLERANCE) {
                    frameBuffer.clear()
                    predHistory.clear()
                    frameCount  = 0
                    noHandCount = 0
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    private fun runInference() {
        val padded = padSequence()
        val input  = Array(1) { Array(sequenceLength) { FloatArray(FEATURE_DIM) } }
        for (t in 0 until sequenceLength) {
            for (f in 0 until FEATURE_DIM) {
                input[0][t][f] = padded[t * FEATURE_DIM + f]
            }
        }

        val output = Array(1) { FloatArray(actions.size) }
        interpreter?.run(input, output)

        val probs      = output[0]
        val candidateIndices = probs.indices.filter { idx ->
            val isLetter = actions[idx].startsWith("letter_")
            if (recognitionMode == RecognitionMode.LETTERS) isLetter else !isLetter
        }
        if (candidateIndices.isEmpty()) return

        val maxIdx     = candidateIndices.maxByOrNull { probs[it] } ?: return
        val confidence = probs[maxIdx]
        val predicted  = actions[maxIdx]

        Log.d(TAG, "Inference: $predicted conf=$confidence")

        // Build top 3 predictions and send to UI
        val top3 = candidateIndices
            .sortedByDescending { probs[it] }
            .take(3)
            .map { idx ->
                val label = if (actions[idx].startsWith("letter_"))
                    actions[idx].removePrefix("letter_").uppercase()
                else
                    actions[idx].replace("_", " ")
                Pair(label, probs[idx])
            }
        onTop3Ready?.invoke(top3)

        if (predHistory.size >= 6) predHistory.removeFirst()
        predHistory.addLast(predicted)

        val voted  = predHistory.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: return
        val stable = predHistory.count { it == voted }
        val now    = System.currentTimeMillis()

        if (confidence >= CONFIDENCE_THRESHOLD
            && stable >= STABLE_REQUIRED
            && voted != lastOutput
            && (now - lastOutputTime) > COOLDOWN_MS) {
            lastOutput     = voted
            lastOutputTime = now
            Log.d(TAG, "DETECTED: $voted (conf=$confidence)")

            if (recognitionMode == RecognitionMode.LETTERS && voted.startsWith("letter_")) {
                // Letter → send to word assembly buffer
                val letter = voted.removePrefix("letter_").uppercase()
                onLetterDetected?.invoke(letter)
            } else if (recognitionMode == RecognitionMode.WORDS && !voted.startsWith("letter_")) {
                // Word sign → send directly to conversation thread
                val displayText = voted.replace("_", " ")
                onGestureRecognized?.invoke(displayText)
            }
        }
    }

    private fun extractKeypoints(
        handResult: HandLandmarkerResult?,
        poseResult: PoseLandmarkerResult?
    ): FloatArray? {
        // Need at least one hand detected
        if (handResult == null || handResult.landmarks().isEmpty()) return null

        val lhRaw   = FloatArray(63)
        val rhRaw   = FloatArray(63)
        val poseRaw = FloatArray(12)

        // Extract hand landmarks — wrist-centered
        val handedness = handResult.handedness()
        val landmarks  = handResult.landmarks()

        for (i in landmarks.indices) {
            val lms   = landmarks[i]
            val label = if (i < handedness.size) handedness[i][0].categoryName() else "Right"

            val wristX = lms[0].x()
            val wristY = lms[0].y()
            val wristZ = lms[0].z()

            // SCALE NORMALIZATION (must match dataset_builder.py + live_recognizer.py EXACTLY):
            // hand_size = distance from wrist(0) to middle-fingertip(12), measured on the
            // WRIST-CENTERED coordinates (i.e. relative to the wrist). Then divide every
            // wrist-centered landmark by that size. Guard against zero size.
            val mid = lms[12]
            val mcX = mid.x() - wristX
            val mcY = mid.y() - wristY
            val mcZ = mid.z() - wristZ
            var handSize = Math.sqrt((mcX * mcX + mcY * mcY + mcZ * mcZ).toDouble()).toFloat()
            if (handSize <= 1e-6f) handSize = 1.0f   // avoid divide-by-zero; leaves centered coords unscaled

            if (label == "Left") {
                for (j in lms.indices) {
                    lhRaw[j * 3]     = (lms[j].x() - wristX) / handSize
                    lhRaw[j * 3 + 1] = (lms[j].y() - wristY) / handSize
                    lhRaw[j * 3 + 2] = (lms[j].z() - wristZ) / handSize
                }
            } else {
                for (j in lms.indices) {
                    rhRaw[j * 3]     = (lms[j].x() - wristX) / handSize
                    rhRaw[j * 3 + 1] = (lms[j].y() - wristY) / handSize
                    rhRaw[j * 3 + 2] = (lms[j].z() - wristZ) / handSize
                }
            }
        }

        // Extract pose landmarks — nose-centered
        // Matches Python: nose(0), left_shoulder(11), right_shoulder(12), left_elbow(13)
        if (poseResult != null && poseResult.landmarks().isNotEmpty()) {
            val poseLms = poseResult.landmarks()[0]
            if (poseLms.size > 13) {
                val noseX = poseLms[0].x()
                val noseY = poseLms[0].y()
                val noseZ = poseLms[0].z()

                val poseIndices = listOf(0, 11, 12, 13) // nose, l_shoulder, r_shoulder, l_elbow
                for ((idx, poseIdx) in poseIndices.withIndex()) {
                    poseRaw[idx * 3]     = poseLms[poseIdx].x() - noseX
                    poseRaw[idx * 3 + 1] = poseLms[poseIdx].y() - noseY
                    poseRaw[idx * 3 + 2] = poseLms[poseIdx].z() - noseZ
                }
            }
        }

        return lhRaw + rhRaw + poseRaw
    }

    private fun padSequence(): FloatArray {
        val flat = FloatArray(sequenceLength * FEATURE_DIM)
        val buf  = frameBuffer.toList()
        val n    = buf.size

        if (n == 0) return flat

        val padCount = sequenceLength - n
        val first    = buf[0]

        for (t in 0 until padCount) {
            first.copyInto(flat, t * FEATURE_DIM)
        }
        for (t in 0 until n) {
            buf[t].copyInto(flat, (padCount + t) * FEATURE_DIM)
        }
        return flat
    }

    fun clearBuffer() {
        frameBuffer.clear()
        predHistory.clear()
        frameCount    = 0
        poseFrameCounter = 0
        handFrameCounter = 0
        noHandCount   = 0
        cachedPoseResult = null
        cachedHandResult = null
        lastOutput    = ""
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun flipBitmapHorizontal(bitmap: Bitmap): Bitmap {
        val matrix = Matrix().apply { postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_FILE)
        val stream = FileInputStream(fd.fileDescriptor)
        return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    fun close() {
        interpreter?.close()
        handLandmarker?.close()
        poseLandmarker?.close()
    }
}

operator fun FloatArray.plus(other: FloatArray): FloatArray {
    val result = FloatArray(this.size + other.size)
    this.copyInto(result, 0)
    other.copyInto(result, this.size)
    return result
}
