package com.sdxlnpu.app;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.provider.Settings;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import android.content.Intent;
import android.content.SharedPreferences;
import android.view.Menu;
import android.view.MenuItem;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.tabs.TabLayout;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SDXL Lightning on Qualcomm NPU — standalone phone generation.
 *
 * Calls phone_generate.py through a regular shell and configurable Python command,
 * parses stdout for progress,
 * loads the resulting PNG and displays it. No JNI, no PC required.
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "SDXLNPU";
    private static final String MODEL_FAMILY_SDXL = "sdxl";
    private static final String MODEL_FAMILY_WAN21 = "wan21";
    private static final String PREF_KEY_SELECTED_MODEL_FAMILY = "selected_model_family";
    private static final String PREF_PREFIX_PROMPT = "last_prompt_";
    private static final String PREF_PREFIX_NEG_PROMPT = "last_neg_prompt_";
    private static final String PREF_PREFIX_SEED = "last_seed_";
    private static final String PREF_PREFIX_STEPS = "last_steps_";
    private static final String PREF_PREFIX_CFG_X10 = "last_cfg_x10_";
    private static final String PREF_PREFIX_WIDTH = "last_width_";
    private static final String PREF_PREFIX_HEIGHT = "last_height_";
    private static final String PREF_PREFIX_CONTRAST = "last_contrast_stretch_";
    private static final String PREF_PREFIX_LIVE_PREVIEW = "last_live_preview_";
    private static final String PREF_PREFIX_PROGRESSIVE_CFG = "last_progressive_cfg_";
    private static final String LEGACY_BASE_DIR = SettingsActivity.LEGACY_BASE_DIR;
    private static final String[] TERMUX_PRIVATE_PREFIXES = new String[] {
        "/data/data/com.termux/",
        "/data/user/0/com.termux/",
    };

    private String BASE_DIR;
    private String ROOT_BASE_DIR;
    private String GEN_SCRIPT;
    private String OUTPUT_DIR;
    private String PYTHON;

    private EditText promptInput;
    private EditText negPromptInput;
    private EditText widthInput;
    private EditText heightInput;
    private EditText seedInput;
    private TabLayout modelFamilyTabs;
    private Spinner sizePresetSpinner;
    private SeekBar stepsSeekBar;
    private TextView stepsLabel;
    private SeekBar cfgSeekBar;
    private TextView cfgLabel;
    private CheckBox contrastStretch;
    private CheckBox livePreview;
    private CheckBox progressiveCfg;
    private MaterialButton generateButton;
    private MaterialButton saveButton;
    private MaterialButton stopButton;
    private MaterialButton copyErrorButton;
    private ProgressBar progressBar;
    private TextView statusText;
    private TextView timingText;
    private ImageView imagePreview;

    private ExecutorService executor;
    private ExecutorService previewExecutor;
    private Handler mainHandler;
    private final Object bitmapLock = new Object();
    private Bitmap currentBitmap;
    private Bitmap displayedBitmap;
    private volatile Process currentProcess;
    private volatile boolean isGenerating = false;
    private static final String PREVIEW_PNG_NAME = "preview_current.png";
    private static final String APK_QNN_PERF_PROFILE = "burst";
    // v0.4.7 line keeps the stability-first warmup policy disabled
    // until it is re-validated on a real phone without RAM-pressure regressions.
    private static final boolean APK_BACKGROUND_PREWARM_ENABLED = false;
    private static final boolean APK_AGGRESSIVE_CONTEXT_PRIMING_ENABLED = false;
    private static final int PREVIEW_DISPLAY_MAX_EDGE = 640;
    private static final int FINAL_DISPLAY_MAX_EDGE = 960;
    private volatile Boolean rootShellAvailable = null;
    private volatile Process prewarmProcess = null;
    private volatile boolean prewarmStartQueued = false;
    private Runnable prewarmKillRunnable = null;
    private Runnable activePreviewPoller = null;
    private volatile boolean previewDecodeInFlight = false;
    private static final long PREWARM_KILL_DELAY_MS = 30_000;
    private boolean updatingSizePresetUi = false;
    private String selectedModelFamily = MODEL_FAMILY_SDXL;

    private static final int[][] SDXL_SIZE_PRESET_DIMENSIONS = new int[][] {
        {1024, 1024},
        {1216, 832},
        {832, 1216},
        {1344, 768},
        {768, 1344},
    };

    private static final int[][] WAN_SIZE_PRESET_DIMENSIONS = new int[][] {
        {832, 480},
        {1280, 720},
    };

    private final TextWatcher sizeInputWatcher = new TextWatcher() {
        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {}

        @Override
        public void afterTextChanged(Editable s) {
            if (!updatingSizePresetUi) {
                selectSizePresetForCurrentInputs();
            }
        }
    };

    // Patterns for parsing generate.py stdout
    private static final Pattern PAT_CLIP  = Pattern.compile("^\\[CLIP (cond|uncond)\\]\\s+L=(\\d+)ms G=(\\d+)ms\\s*$");
    private static final Pattern PAT_UNET  = Pattern.compile("^\\s*\\[UNet (\\d+)/(\\d+)\\][^\\n]*?\\s(\\d+)ms(?:\\s|$)");
    private static final Pattern PAT_PREV  = Pattern.compile("^\\s*\\[PREVIEW step (\\d+)/(\\d+)\\]\\s+(?:[A-Z]+(?:\\s+[A-Z]+)?\\s+)?(\\d+)ms\\s*$");
    private static final Pattern PAT_TAESD_WARNING = Pattern.compile("^TAESD_WARNING:\\s*(.+?)\\s*$");
    private static final Pattern PAT_TEMP_LINE = Pattern.compile("^\\s*\\[TEMP\\]\\s+(.+)$");
    private static final Pattern PAT_TEMP_ITEM = Pattern.compile("(CPU|GPU|NPU)=([\\d.]+)°C");
    private static final Pattern PAT_VAE   = Pattern.compile("^\\[VAE\\]\\s+(\\d+)ms\\s*$");
    private static final Pattern PAT_SAVED = Pattern.compile("Saved:\\s+(.+\\.png)");
    private static final Pattern PAT_TOTAL = Pattern.compile("Total:\\s+([\\d.]+)s");
    private volatile String latestStageStatus = "";
    private volatile String latestWarningStatus = "";
    private volatile String latestTempStatus = "";
    private volatile String lastErrorDetails = "";
    private volatile int latestProgress = 0;
    private volatile boolean storageAccessPromptShown = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        promptInput = findViewById(R.id.promptInput);
        negPromptInput = findViewById(R.id.negPromptInput);
        widthInput = findViewById(R.id.widthInput);
        heightInput = findViewById(R.id.heightInput);
        seedInput = findViewById(R.id.seedInput);
        modelFamilyTabs = findViewById(R.id.modelFamilyTabs);
        sizePresetSpinner = findViewById(R.id.sizePresetSpinner);
        stepsSeekBar = findViewById(R.id.stepsSeekBar);
        stepsLabel = findViewById(R.id.stepsLabel);
        cfgSeekBar = findViewById(R.id.cfgSeekBar);
        cfgLabel = findViewById(R.id.cfgLabel);
        contrastStretch = findViewById(R.id.contrastStretch);
        livePreview     = findViewById(R.id.livePreview);
        progressiveCfg  = findViewById(R.id.progressiveCfg);
        generateButton  = findViewById(R.id.generateButton);
        saveButton = findViewById(R.id.saveButton);
        stopButton = findViewById(R.id.stopButton);
        copyErrorButton = findViewById(R.id.copyErrorButton);
        progressBar = findViewById(R.id.progressBar);
        statusText = findViewById(R.id.statusText);
        timingText = findViewById(R.id.timingText);
        imagePreview = findViewById(R.id.imagePreview);

        executor = Executors.newSingleThreadExecutor();
        previewExecutor = Executors.newSingleThreadExecutor();
        mainHandler = new Handler(Looper.getMainLooper());

        loadSettings();
        configureModelFamilyTabs();
        configureSizePresetSpinner();
        widthInput.addTextChangedListener(sizeInputWatcher);
        heightInput.addTextChangedListener(sizeInputWatcher);
        selectSizePresetForCurrentInputs();

        stepsSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                stepsLabel.setText(String.format(Locale.US, "Steps: %d", progress));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        cfgSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                cfgLabel.setText(String.format(Locale.US, "CFG: %.1f", progress / 10f));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        generateButton.setOnClickListener(v -> startGeneration());
        saveButton.setOnClickListener(v -> saveImage());
        stopButton.setOnClickListener(v -> stopGeneration());
        copyErrorButton.setOnClickListener(v -> copyLastErrorToClipboard());

        // Check prerequisites
        checkPrerequisites();

        // Start prewarm in background
        if (APK_BACKGROUND_PREWARM_ENABLED) {
            startPrewarm();
        }

        // Pre-extract py_runtime in background so it's ready when generation starts
        executor.submit(() -> {
            try {
                RuntimeBootstrap.ensurePyRuntimeExtracted(this);
            } catch (Exception e) {
                Log.w(TAG, "Background py_runtime extraction failed: " + e.getMessage());
            }
        });
    }

    @Override
    protected void onStart() {
        super.onStart();
        // Cancel any pending prewarm kill from minimize
        cancelScheduledPrewarmKill();
        // Restart prewarm if it was killed
        if (APK_BACKGROUND_PREWARM_ENABLED && (prewarmProcess == null || !prewarmProcess.isAlive())) {
            startPrewarm();
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        schedulePrewarmKill("app moved to background");
    }

    private void cancelScheduledPrewarmKill() {
        if (prewarmKillRunnable != null) {
            mainHandler.removeCallbacks(prewarmKillRunnable);
            prewarmKillRunnable = null;
        }
    }

    private void killPrewarmNow(String reason) {
        cancelScheduledPrewarmKill();
        Process process = prewarmProcess;
        if (process == null) {
            return;
        }
        try {
            if (process.isAlive()) {
                Log.i(TAG, "prewarm: stopping (" + reason + ")");
                process.destroyForcibly();
            }
        } catch (Exception e) {
            Log.w(TAG, "prewarm: failed to stop cleanly (" + reason + ")", e);
        } finally {
            prewarmProcess = null;
        }
    }

    private void schedulePrewarmKill(String reason) {
        cancelScheduledPrewarmKill();
        if (!MODEL_FAMILY_SDXL.equals(getSelectedModelFamily()) || isGenerating) {
            return;
        }
        Process process = prewarmProcess;
        if (process == null || !process.isAlive()) {
            prewarmProcess = null;
            return;
        }
        prewarmKillRunnable = () -> {
            Process current = prewarmProcess;
            if (current != null && current.isAlive() && !isGenerating) {
                Log.i(TAG, "prewarm: idle timeout reached (" + reason + ")");
                current.destroyForcibly();
            }
            prewarmProcess = null;
            prewarmKillRunnable = null;
        };
        mainHandler.postDelayed(prewarmKillRunnable, PREWARM_KILL_DELAY_MS);
        Log.i(TAG, "prewarm: scheduled release in " + PREWARM_KILL_DELAY_MS + "ms (" + reason + ")");
    }

    private void startPrewarm() {
        if (!APK_BACKGROUND_PREWARM_ENABLED) {
            Log.i(TAG, "startPrewarm: disabled for stability-first APK line");
            return;
        }
        if (!MODEL_FAMILY_SDXL.equals(getSelectedModelFamily())) {
            Log.i(TAG, "startPrewarm: skip for WAN/basic-debug mode");
            return;
        }
        Process running = prewarmProcess;
        if (running != null && running.isAlive()) {
            cancelScheduledPrewarmKill();
            Log.i(TAG, "startPrewarm: already running");
            return;
        }
        if (prewarmStartQueued) {
            Log.i(TAG, "startPrewarm: helper launch already queued");
            return;
        }
        cancelScheduledPrewarmKill();
        prewarmStartQueued = true;
        executor.execute(() -> {
            try {
                if (isGenerating) {
                    Log.i(TAG, "startPrewarm: skip because generation is already active");
                    return;
                }
                Log.i(TAG, "startPrewarm: begin");
                String activeBaseDir = resolveActiveBaseDir(MODEL_FAMILY_SDXL);
                ExecutionPlan plan = resolveExecutionPlan(activeBaseDir);
                Log.i(TAG, "startPrewarm: plan root=" + plan.useRootShell + ", python=" + plan.pythonCommand);
                File bundledPayload = getBundledRuntimePayloadDirOrNull();
                String generatorScript = resolveGeneratorScriptPath(bundledPayload);
                File runtimeRoot = ensureDir(new File(getCacheDir(), "sdxl_runtime"));
                File runtimeWorkDir = ensureDir(new File(runtimeRoot, "work"));
                final String runtimeWorkDirPath = runtimeWorkDir.getAbsolutePath();

                StringBuilder script = new StringBuilder();
                appendShellEnvironment(script);
                script.append("export MODEL_TO_NPU_BASE=\"").append(shellEscape(activeBaseDir)).append("\"\n");
                script.append("export SDXL_QNN_BASE=\"").append(shellEscape(activeBaseDir)).append("\"\n");
                script.append("export SDXL_QNN_WORK_DIR=\"").append(shellEscape(runtimeWorkDirPath)).append("\"\n");
                script.append("export PYTHONDONTWRITEBYTECODE=1\n");
                script.append("export SDXL_QNN_USE_MMAP=1\n");
                script.append("export SDXL_QNN_LOG_LEVEL=warn\n");
                script.append("export SDXL_QNN_PERF_PROFILE=").append(APK_QNN_PERF_PROFILE).append("\n");
                script.append("export SDXL_QNN_SHARED_SERVER=0\n");
                script.append("export SDXL_QNN_PRESTAGE_RUNTIME=1\n");
                boolean bundledQnnConfigReady = appendBundledRuntimeEnvironment(script, bundledPayload);
                if (!bundledQnnConfigReady) {
                    script.append("if [ -f \"").append(shellEscape(activeBaseDir)).append("/htp_backend_extensions_lightning.json\" ] && [ -f \"")
                        .append(shellEscape(activeBaseDir)).append("/lib/libQnnHtpNetRunExtensions.so\" ]; then\n");
                    script.append("  export SDXL_QNN_CONFIG_FILE=\"").append(shellEscape(activeBaseDir))
                        .append("/htp_backend_extensions_lightning.json\"\n");
                    script.append("fi\n");
                }
                script.append(String.format(Locale.US,
                    "export LD_LIBRARY_PATH=\"%s/lib:%s/bin:%s/model:$LD_LIBRARY_PATH\"\n",
                    shellEscape(activeBaseDir), shellEscape(activeBaseDir), shellEscape(activeBaseDir)));
                script.append(String.format(Locale.US,
                    "export ADSP_LIBRARY_PATH=\"%s/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp\"\n",
                    shellEscape(activeBaseDir)));
                script.append("cd \"").append(shellEscape(activeBaseDir)).append("\"\n");
                script.append("exec \"").append(shellEscape(plan.pythonCommand))
                    .append("\" \"").append(shellEscape(generatorScript))
                    .append("\" --prewarm 2>&1\n");

                ProcessBuilder pb;
                if (plan.useRootShell) {
                    String su = findAvailableSuOrNull();
                    if (su == null) return;
                    pb = new ProcessBuilder(su, "--mount-master");
                } else {
                    pb = new ProcessBuilder("/system/bin/sh");
                }
                pb.redirectErrorStream(true);
                Process process = pb.start();

                try (OutputStream os = process.getOutputStream()) {
                    os.write(script.toString().getBytes("UTF-8"));
                    os.flush();
                }

                prewarmProcess = process;
                Log.i(TAG, "startPrewarm: process started");

                // Read output until PREWARM_READY or process dies
                boolean ready = false;
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        Log.i(TAG, "prewarm> " + line);
                        if (line.contains("PREWARM_READY")) {
                            ready = true;
                            break;
                        }
                    }
                }
                if (!ready && !process.isAlive()) {
                    prewarmProcess = null;
                }
            } catch (Exception e) {
                Log.w(TAG, "startPrewarm failed", e);
                prewarmProcess = null;
                // Prewarm is best-effort — don't crash the app
            } finally {
                prewarmStartQueued = false;
            }
        });
    }

    private static boolean isKnownModelFamily(String modelFamily) {
        return MODEL_FAMILY_SDXL.equals(modelFamily) || MODEL_FAMILY_WAN21.equals(modelFamily);
    }

    private static String prefKeyForFamily(String prefix, String modelFamily) {
        return prefix + modelFamily;
    }

    private void saveGenerationSettingsForFamily(SharedPreferences.Editor editor, String modelFamily) {
        editor.putString(prefKeyForFamily(PREF_PREFIX_PROMPT, modelFamily), promptInput.getText().toString());
        editor.putString(prefKeyForFamily(PREF_PREFIX_NEG_PROMPT, modelFamily), negPromptInput.getText().toString());
        editor.putString(prefKeyForFamily(PREF_PREFIX_SEED, modelFamily), seedInput.getText().toString());
        editor.putInt(prefKeyForFamily(PREF_PREFIX_STEPS, modelFamily), stepsSeekBar.getProgress());
        editor.putInt(prefKeyForFamily(PREF_PREFIX_CFG_X10, modelFamily), cfgSeekBar.getProgress());
        editor.putString(prefKeyForFamily(PREF_PREFIX_WIDTH, modelFamily), widthInput.getText().toString());
        editor.putString(prefKeyForFamily(PREF_PREFIX_HEIGHT, modelFamily), heightInput.getText().toString());
        editor.putBoolean(prefKeyForFamily(PREF_PREFIX_CONTRAST, modelFamily), contrastStretch.isChecked());
        editor.putBoolean(prefKeyForFamily(PREF_PREFIX_LIVE_PREVIEW, modelFamily), livePreview.isChecked());
        editor.putBoolean(prefKeyForFamily(PREF_PREFIX_PROGRESSIVE_CFG, modelFamily), progressiveCfg.isChecked());
    }

    private void restoreGenerationSettingsForFamily(
            SharedPreferences prefs,
            String modelFamily,
            boolean allowLegacyFallback) {
        final boolean wanMode = MODEL_FAMILY_WAN21.equals(modelFamily);
        final int defaultSteps = 8;
        final int defaultCfgX10 = wanMode ? 10 : 35;
        final String defaultWidth = wanMode ? "832" : "1024";
        final String defaultHeight = wanMode ? "480" : "1024";

        String prompt = prefs.getString(prefKeyForFamily(PREF_PREFIX_PROMPT, modelFamily), null);
        String negativePrompt = prefs.getString(prefKeyForFamily(PREF_PREFIX_NEG_PROMPT, modelFamily), null);
        String seed = prefs.getString(prefKeyForFamily(PREF_PREFIX_SEED, modelFamily), null);
        Integer steps = prefs.contains(prefKeyForFamily(PREF_PREFIX_STEPS, modelFamily))
            ? prefs.getInt(prefKeyForFamily(PREF_PREFIX_STEPS, modelFamily), defaultSteps)
            : null;
        Integer cfgX10 = prefs.contains(prefKeyForFamily(PREF_PREFIX_CFG_X10, modelFamily))
            ? prefs.getInt(prefKeyForFamily(PREF_PREFIX_CFG_X10, modelFamily), defaultCfgX10)
            : null;
        String width = prefs.getString(prefKeyForFamily(PREF_PREFIX_WIDTH, modelFamily), null);
        String height = prefs.getString(prefKeyForFamily(PREF_PREFIX_HEIGHT, modelFamily), null);
        Boolean contrast = prefs.contains(prefKeyForFamily(PREF_PREFIX_CONTRAST, modelFamily))
            ? prefs.getBoolean(prefKeyForFamily(PREF_PREFIX_CONTRAST, modelFamily), true)
            : null;
        Boolean preview = prefs.contains(prefKeyForFamily(PREF_PREFIX_LIVE_PREVIEW, modelFamily))
            ? prefs.getBoolean(prefKeyForFamily(PREF_PREFIX_LIVE_PREVIEW, modelFamily), false)
            : null;
        Boolean progCfg = prefs.contains(prefKeyForFamily(PREF_PREFIX_PROGRESSIVE_CFG, modelFamily))
            ? prefs.getBoolean(prefKeyForFamily(PREF_PREFIX_PROGRESSIVE_CFG, modelFamily), false)
            : null;

        if (allowLegacyFallback && MODEL_FAMILY_SDXL.equals(modelFamily)) {
            if (prompt == null) {
                prompt = prefs.getString("last_prompt", "");
            }
            if (negativePrompt == null) {
                negativePrompt = prefs.getString("last_neg_prompt", "");
            }
            if (seed == null) {
                seed = prefs.getString("last_seed", "");
            }
            if (steps == null) {
                steps = prefs.getInt("last_steps", defaultSteps);
            }
            if (cfgX10 == null) {
                cfgX10 = prefs.getInt("last_cfg_x10", defaultCfgX10);
            }
            if (width == null) {
                width = prefs.getString("last_width", defaultWidth);
            }
            if (height == null) {
                height = prefs.getString("last_height", defaultHeight);
            }
            if (contrast == null) {
                contrast = prefs.getBoolean("last_contrast_stretch", true);
            }
            if (preview == null) {
                preview = prefs.getBoolean("last_live_preview", false);
            }
            if (progCfg == null) {
                progCfg = prefs.getBoolean("last_progressive_cfg", false);
            }
        }

        promptInput.setText(prompt != null ? prompt : "");
        negPromptInput.setText(negativePrompt != null ? negativePrompt : "");
        seedInput.setText(seed != null ? seed : "");
        stepsSeekBar.setProgress(steps != null ? steps : defaultSteps);
        stepsLabel.setText(String.format(Locale.US, "Steps: %d", stepsSeekBar.getProgress()));
        cfgSeekBar.setProgress(cfgX10 != null ? cfgX10 : defaultCfgX10);
        cfgLabel.setText(String.format(Locale.US, "CFG: %.1f", cfgSeekBar.getProgress() / 10f));
        widthInput.setText(width != null ? width : defaultWidth);
        heightInput.setText(height != null ? height : defaultHeight);
        contrastStretch.setChecked(contrast != null ? contrast : true);
        livePreview.setChecked(preview != null ? preview : false);
        progressiveCfg.setChecked(progCfg != null ? progCfg : false);

        applyModelFamilyDefaults(wanMode);
    }

    private void configureModelFamilyTabs() {
        if (modelFamilyTabs == null) {
            return;
        }
        modelFamilyTabs.removeAllTabs();
        modelFamilyTabs.addTab(
            modelFamilyTabs.newTab().setText(getString(R.string.model_family_tab_sdxl)),
            MODEL_FAMILY_SDXL.equals(selectedModelFamily)
        );
        modelFamilyTabs.addTab(
            modelFamilyTabs.newTab().setText(getString(R.string.model_family_tab_wan)),
            MODEL_FAMILY_WAN21.equals(selectedModelFamily)
        );
        modelFamilyTabs.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                if (tab == null) {
                    return;
                }
                String modelFamily = tab.getPosition() == 1
                    ? MODEL_FAMILY_WAN21
                    : MODEL_FAMILY_SDXL;
                switchModelFamily(modelFamily);
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {}

            @Override
            public void onTabReselected(TabLayout.Tab tab) {}
        });
        syncModelFamilyTabSelection();
    }

    private void syncModelFamilyTabSelection() {
        if (modelFamilyTabs == null || modelFamilyTabs.getTabCount() < 2) {
            return;
        }
        int tabIndex = MODEL_FAMILY_WAN21.equals(selectedModelFamily) ? 1 : 0;
        TabLayout.Tab tab = modelFamilyTabs.getTabAt(tabIndex);
        if (tab != null && !tab.isSelected()) {
            tab.select();
        }
    }

    private void switchModelFamily(String modelFamily) {
        String normalized = isKnownModelFamily(modelFamily) ? modelFamily : MODEL_FAMILY_SDXL;
        if (normalized.equals(selectedModelFamily)) {
            return;
        }

        String previous = selectedModelFamily;
        SharedPreferences prefs = getSharedPreferences(SettingsActivity.PREFS_NAME, MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();
        saveGenerationSettingsForFamily(editor, previous);
        editor.putString(PREF_KEY_SELECTED_MODEL_FAMILY, normalized);
        editor.apply();

        selectedModelFamily = normalized;
        restoreGenerationSettingsForFamily(prefs, selectedModelFamily, false);
        refreshSizePresetSpinnerItems();
        selectSizePresetForCurrentInputs();

        if (MODEL_FAMILY_WAN21.equals(selectedModelFamily)) {
            killPrewarmNow("switch to WAN/basic-debug mode");
        } else if (APK_BACKGROUND_PREWARM_ENABLED) {
            startPrewarm();
        }

        checkPrerequisites();
    }

    private void loadSettings() {
        SharedPreferences prefs = getSharedPreferences(
            SettingsActivity.PREFS_NAME, MODE_PRIVATE);
        String detectedBaseDir = SettingsActivity.detectDefaultBaseDir();
        String detectedPython = SettingsActivity.detectDefaultPython(detectedBaseDir);
        BASE_DIR = prefs.getString(SettingsActivity.KEY_BASE_DIR,
            detectedBaseDir);
        ROOT_BASE_DIR = prefs.getString(SettingsActivity.KEY_ROOT_BASE_DIR,
            SettingsActivity.DEFAULT_ROOT_BASE_DIR);
        PYTHON = prefs.getString(SettingsActivity.KEY_PYTHON_PATH,
            detectedPython);
        GEN_SCRIPT = BASE_DIR + "/phone_gen/generate.py";
        OUTPUT_DIR = BASE_DIR + "/outputs";

        String storedFamily = prefs.getString(PREF_KEY_SELECTED_MODEL_FAMILY, MODEL_FAMILY_SDXL);
        selectedModelFamily = isKnownModelFamily(storedFamily) ? storedFamily : MODEL_FAMILY_SDXL;
        restoreGenerationSettingsForFamily(prefs, selectedModelFamily, true);
        syncModelFamilyTabSelection();
    }

    private void configureSizePresetSpinner() {
        if (sizePresetSpinner == null) {
            return;
        }
        refreshSizePresetSpinnerItems();
        sizePresetSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (updatingSizePresetUi || position < 0) {
                    return;
                }
                int[][] presets = getActiveSizePresetDimensions();
                if (position >= presets.length) {
                    return;
                }
                int[] preset = presets[position];
                updatingSizePresetUi = true;
                widthInput.setText(String.valueOf(preset[0]));
                heightInput.setText(String.valueOf(preset[1]));
                updatingSizePresetUi = false;
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });
    }

    private static int parsePositiveInt(EditText input) {
        if (input == null || input.getText() == null) {
            return 0;
        }
        String value = input.getText().toString().trim();
        if (value.isEmpty()) {
            return 0;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }

    private int findNearestSizePresetIndex(int width, int height) {
        int[][] presets = getActiveSizePresetDimensions();
        if (presets.length == 0) {
            return 0;
        }
        if (width <= 0 || height <= 0) {
            return 0;
        }
        int bestIndex = 0;
        long bestScore = Long.MAX_VALUE;
        for (int i = 0; i < presets.length; i++) {
            int[] preset = presets[i];
            long dw = (long) preset[0] - width;
            long dh = (long) preset[1] - height;
            long score = dw * dw + dh * dh;
            if (score < bestScore) {
                bestScore = score;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private void applySizePresetByIndex(int index) {
        int[][] presets = getActiveSizePresetDimensions();
        if (presets.length == 0) {
            return;
        }
        int safeIndex = Math.max(0, Math.min(index, presets.length - 1));
        int[] preset = presets[safeIndex];
        updatingSizePresetUi = true;
        widthInput.setText(String.valueOf(preset[0]));
        heightInput.setText(String.valueOf(preset[1]));
        if (sizePresetSpinner != null && sizePresetSpinner.getSelectedItemPosition() != safeIndex) {
            sizePresetSpinner.setSelection(safeIndex, false);
        }
        updatingSizePresetUi = false;
    }

    private void selectSizePresetForCurrentInputs() {
        if (sizePresetSpinner == null) {
            return;
        }
        int width = parsePositiveInt(widthInput);
        int height = parsePositiveInt(heightInput);
        int selection = findNearestSizePresetIndex(width, height);
        int[][] presets = getActiveSizePresetDimensions();
        if (presets.length == 0) {
            return;
        }
        int[] preset = presets[selection];
        if (preset[0] != width || preset[1] != height) {
            applySizePresetByIndex(selection);
            return;
        }
        if (sizePresetSpinner.getSelectedItemPosition() != selection) {
            updatingSizePresetUi = true;
            sizePresetSpinner.setSelection(selection, false);
            updatingSizePresetUi = false;
        }
    }

    private void saveGenerationSettings() {
        SharedPreferences prefs = getSharedPreferences(
            SettingsActivity.PREFS_NAME, MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();
        String modelFamily = getSelectedModelFamily();
        saveGenerationSettingsForFamily(editor, modelFamily);
        editor.putString(PREF_KEY_SELECTED_MODEL_FAMILY, modelFamily);

        // Backward-compatibility keys for older APK lines (SDXL defaults).
        if (MODEL_FAMILY_SDXL.equals(modelFamily)) {
            editor.putString("last_prompt", promptInput.getText().toString());
            editor.putString("last_neg_prompt", negPromptInput.getText().toString());
            editor.putString("last_seed", seedInput.getText().toString());
            editor.putInt("last_steps", stepsSeekBar.getProgress());
            editor.putInt("last_cfg_x10", cfgSeekBar.getProgress());
            editor.putString("last_width", widthInput.getText().toString());
            editor.putString("last_height", heightInput.getText().toString());
            editor.putBoolean("last_contrast_stretch", contrastStretch.isChecked());
            editor.putBoolean("last_live_preview", livePreview.isChecked());
            editor.putBoolean("last_progressive_cfg", progressiveCfg.isChecked());
        }
        editor.apply();
    }

    @Override
    protected void onPause() {
        super.onPause();
        saveGenerationSettings();
    }

    private String getSelectedModelFamily() {
        return selectedModelFamily;
    }

    private void applyModelFamilyDefaults(boolean wanMode) {
        String widthText = widthInput.getText() != null ? widthInput.getText().toString().trim() : "";
        String heightText = heightInput.getText() != null ? heightInput.getText().toString().trim() : "";
        if (wanMode) {
            if (widthText.isEmpty() || heightText.isEmpty()
                || ("1024".equals(widthText) && "1024".equals(heightText))) {
                widthInput.setText("832");
                heightInput.setText("480");
            }
            livePreview.setChecked(false);
        } else if (("832".equals(widthText) && "480".equals(heightText))
                || (widthText.isEmpty() && heightText.isEmpty())) {
            widthInput.setText("1024");
            heightInput.setText("1024");
        }
    }

    private int[][] getActiveSizePresetDimensions() {
        return MODEL_FAMILY_WAN21.equals(getSelectedModelFamily())
            ? WAN_SIZE_PRESET_DIMENSIONS
            : SDXL_SIZE_PRESET_DIMENSIONS;
    }

    private void refreshSizePresetSpinnerItems() {
        if (sizePresetSpinner == null) {
            return;
        }
        int[][] presets = getActiveSizePresetDimensions();
        String[] labels = new String[presets.length];
        for (int i = 0; i < presets.length; i++) {
            int[] preset = presets[i];
            labels[i] = getString(R.string.size_preset_format, preset[0], preset[1]);
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<>(
            this,
            android.R.layout.simple_spinner_item,
            labels
        );
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        sizePresetSpinner.setAdapter(adapter);
    }

    private void cancelPreviewPolling() {
        Runnable poller = activePreviewPoller;
        if (poller != null) {
            mainHandler.removeCallbacks(poller);
            activePreviewPoller = null;
        }
    }

    private void recycleBitmapQuietly(Bitmap bitmap) {
        if (bitmap == null || bitmap.isRecycled()) {
            return;
        }
        try {
            bitmap.recycle();
        } catch (Exception e) {
            Log.w(TAG, "bitmap recycle failed", e);
        }
    }

    private void clearDisplayedImage(boolean recycleSavedBitmap) {
        cancelPreviewPolling();

        Bitmap oldDisplayed;
        Bitmap oldCurrent = null;
        synchronized (bitmapLock) {
            oldDisplayed = displayedBitmap;
            displayedBitmap = null;
            if (recycleSavedBitmap) {
                oldCurrent = currentBitmap;
                currentBitmap = null;
            }
        }

        imagePreview.setImageBitmap(null);

        if (oldDisplayed != null && oldDisplayed != oldCurrent) {
            recycleBitmapQuietly(oldDisplayed);
        }
        if (recycleSavedBitmap && oldCurrent != null) {
            recycleBitmapQuietly(oldCurrent);
        }
    }

    private void showPreviewBitmap(Bitmap bitmap) {
        Bitmap oldDisplayed;
        Bitmap savedBitmap;
        synchronized (bitmapLock) {
            oldDisplayed = displayedBitmap;
            savedBitmap = currentBitmap;
            displayedBitmap = bitmap;
        }

        imagePreview.setImageBitmap(bitmap);

        if (oldDisplayed != null && oldDisplayed != bitmap && oldDisplayed != savedBitmap) {
            recycleBitmapQuietly(oldDisplayed);
        }
    }

    private void showFinalBitmap(Bitmap bitmap) {
        Bitmap oldDisplayed;
        Bitmap oldCurrent;
        synchronized (bitmapLock) {
            oldDisplayed = displayedBitmap;
            oldCurrent = currentBitmap;
            displayedBitmap = bitmap;
            currentBitmap = bitmap;
        }

        imagePreview.setImageBitmap(bitmap);

        if (oldDisplayed != null && oldDisplayed != bitmap && oldDisplayed != oldCurrent) {
            recycleBitmapQuietly(oldDisplayed);
        }
        if (oldCurrent != null && oldCurrent != bitmap) {
            recycleBitmapQuietly(oldCurrent);
        }
    }

    private void decodePreviewBitmapAsync(String previewPath, Runnable owner) {
        ExecutorService localPreviewExecutor = previewExecutor;
        if (localPreviewExecutor == null || previewDecodeInFlight) {
            return;
        }
        previewDecodeInFlight = true;
        try {
            localPreviewExecutor.execute(() -> {
                Bitmap decoded = null;
                try {
                    decoded = decodeBitmapForDisplay(previewPath, true);
                } catch (Exception e) {
                    Log.w(TAG, "preview decode failed", e);
                }

                Bitmap finalDecoded = decoded;
                mainHandler.post(() -> {
                    try {
                        if (finalDecoded != null) {
                            if (isGenerating && activePreviewPoller == owner) {
                                showPreviewBitmap(finalDecoded);
                            } else {
                                recycleBitmapQuietly(finalDecoded);
                            }
                        }
                    } finally {
                        previewDecodeInFlight = false;
                    }
                });
            });
        } catch (Exception e) {
            previewDecodeInFlight = false;
            Log.w(TAG, "preview decode scheduling failed", e);
        }
    }

    private int resolveDisplayDecodeMaxEdge(boolean previewMode) {
        int screenWidth = getResources().getDisplayMetrics().widthPixels;
        int preferredMaxEdge = previewMode ? PREVIEW_DISPLAY_MAX_EDGE : FINAL_DISPLAY_MAX_EDGE;
        return Math.max(512, Math.min(screenWidth, preferredMaxEdge));
    }

    private static int[] constrainBitmapSize(int width, int height, int maxEdge) {
        if (width <= 0 || height <= 0 || maxEdge <= 0) {
            return new int[] {Math.max(1, width), Math.max(1, height)};
        }
        int longestEdge = Math.max(width, height);
        if (longestEdge <= maxEdge) {
            return new int[] {width, height};
        }
        float scale = maxEdge / (float) longestEdge;
        return new int[] {
            Math.max(1, Math.round(width * scale)),
            Math.max(1, Math.round(height * scale))
        };
    }

    private static int calculateInSampleSize(int width, int height, int maxEdge) {
        if (width <= 0 || height <= 0 || maxEdge <= 0) {
            return 1;
        }
        int sample = 1;
        int longestEdge = Math.max(width, height);
        while ((longestEdge / sample) > maxEdge) {
            sample *= 2;
        }
        return Math.max(1, sample);
    }

    private Bitmap decodeBitmapForDisplayFallback(String path, boolean previewMode, int maxEdge) {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(path, bounds);

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = previewMode ? Bitmap.Config.RGB_565 : Bitmap.Config.ARGB_8888;
        options.inDither = previewMode;
        options.inSampleSize = calculateInSampleSize(bounds.outWidth, bounds.outHeight, maxEdge);
        return BitmapFactory.decodeFile(path, options);
    }

    private Bitmap decodeBitmapForDisplay(String path, boolean previewMode) {
        File imageFile = new File(path);
        if (!imageFile.isFile()) {
            return null;
        }

        int maxEdge = resolveDisplayDecodeMaxEdge(previewMode);
        try {
            ImageDecoder.Source source = ImageDecoder.createSource(imageFile);
            return ImageDecoder.decodeBitmap(source, (decoder, info, src) -> {
                Size sourceSize = info.getSize();
                int[] targetSize = constrainBitmapSize(
                    sourceSize.getWidth(),
                    sourceSize.getHeight(),
                    maxEdge
                );
                decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                decoder.setMemorySizePolicy(ImageDecoder.MEMORY_POLICY_LOW_RAM);
                decoder.setMutableRequired(false);
                decoder.setTargetSize(targetSize[0], targetSize[1]);
            });
        } catch (Exception e) {
            Log.w(TAG, "display decode fallback for " + path, e);
            return decodeBitmapForDisplayFallback(path, previewMode, maxEdge);
        }
    }

    private void restoreOrSchedulePrewarm(String reason) {
        if (!APK_BACKGROUND_PREWARM_ENABLED) {
            return;
        }
        if (!MODEL_FAMILY_SDXL.equals(getSelectedModelFamily()) || isGenerating) {
            return;
        }
        Process process = prewarmProcess;
        if (process != null && process.isAlive()) {
            schedulePrewarmKill(reason);
        }
    }

    private boolean appendBundledRuntimeEnvironment(StringBuilder script, File bundledRuntimePayloadDir) {
        if (bundledRuntimePayloadDir == null) {
            return false;
        }

        File bundledNetRun = new File(bundledRuntimePayloadDir, "bin/qnn-net-run");
        File bundledLibDir = new File(bundledRuntimePayloadDir, "lib");
        File bundledServer = new File(bundledRuntimePayloadDir, "bin/qnn-multi-context-server");
        File bundledDaemon = new File(bundledRuntimePayloadDir, "bin/qnn-context-runner");
        File bundledConfig = new File(bundledRuntimePayloadDir, "htp_backend_extensions_lightning.json");
        File bundledExtLib = new File(bundledRuntimePayloadDir, "lib/libQnnHtpNetRunExtensions.so");

        if (bundledNetRun.isFile()) {
            script.append("export SDXL_QNN_NET_RUN=\"")
                .append(shellEscape(bundledNetRun.getAbsolutePath()))
                .append("\"\n");
        }
        if (bundledLibDir.isDirectory()) {
            script.append("export SDXL_QNN_LIB_DIR=\"")
                .append(shellEscape(bundledLibDir.getAbsolutePath()))
                .append("\"\n");
        }
        if (bundledServer.isFile()) {
            script.append("export SDXL_QNN_SERVER_BIN=\"")
                .append(shellEscape(bundledServer.getAbsolutePath()))
                .append("\"\n");
        }
        if (bundledDaemon.isFile()) {
            script.append("export SDXL_QNN_DAEMON_BIN=\"")
                .append(shellEscape(bundledDaemon.getAbsolutePath()))
                .append("\"\n");
        }
        if (bundledConfig.isFile() && bundledExtLib.isFile()) {
            script.append("export SDXL_QNN_CONFIG_FILE=\"")
                .append(shellEscape(bundledConfig.getAbsolutePath()))
                .append("\"\n");
            return true;
        }
        return false;
    }

    private String resolveActiveBaseDir(String modelFamily) {
        if (MODEL_FAMILY_WAN21.equals(modelFamily)) {
            if (BASE_DIR != null && BASE_DIR.contains("wan21") && new File(BASE_DIR).exists()) {
                return BASE_DIR;
            }
            for (String candidate : new String[] {
                SettingsActivity.WAN_LEGACY_BASE_DIR,
                SettingsActivity.WAN_DOWNLOADS_BASE_DIR,
                "/storage/emulated/0/Download/wan21_t2v_qnn",
            }) {
                if (new File(candidate).exists()) {
                    return candidate;
                }
            }
            return SettingsActivity.WAN_DOWNLOADS_BASE_DIR;
        }
        // SDXL: auto-switch to root path if accessible
        if (ROOT_BASE_DIR != null && !ROOT_BASE_DIR.isEmpty() && new File(ROOT_BASE_DIR).exists()) {
            return ROOT_BASE_DIR;
        }
        return BASE_DIR != null && !BASE_DIR.isEmpty()
            ? BASE_DIR
            : SettingsActivity.detectDefaultBaseDir();
    }

    private String resolveConfiguredBaseDir(String modelFamily) {
        if (MODEL_FAMILY_WAN21.equals(modelFamily)) {
            return BASE_DIR != null && !BASE_DIR.isEmpty()
                ? BASE_DIR
                : SettingsActivity.WAN_DOWNLOADS_BASE_DIR;
        }
        return BASE_DIR != null && !BASE_DIR.isEmpty()
            ? BASE_DIR
            : SettingsActivity.detectDefaultBaseDir();
    }

    private String buildBaseRedirectWarning(String modelFamily, String configuredBaseDir, String activeBaseDir) {
        if (!MODEL_FAMILY_SDXL.equals(modelFamily)) {
            return null;
        }
        if (configuredBaseDir == null || configuredBaseDir.isEmpty()
                || activeBaseDir == null || activeBaseDir.isEmpty()) {
            return null;
        }
        if (configuredBaseDir.equals(activeBaseDir)) {
            return null;
        }
        if (ROOT_BASE_DIR != null && !ROOT_BASE_DIR.isEmpty() && ROOT_BASE_DIR.equals(activeBaseDir)) {
            return "SDXL base автоматически переадресован на legacy/root путь: "
                + activeBaseDir + " (вместо " + configuredBaseDir + ")";
        }
        return null;
    }

    private boolean isLegacyBaseDir(String baseDir) {
        return baseDir != null && (
            baseDir.startsWith(SettingsActivity.LEGACY_BASE_DIR)
                || baseDir.startsWith(SettingsActivity.WAN_LEGACY_BASE_DIR)
        );
    }

    private void checkPrerequisites() {
        clearErrorState();
        String modelFamily = getSelectedModelFamily();
        String activeBaseDir = resolveActiveBaseDir(modelFamily);
        String configuredBaseDir = resolveConfiguredBaseDir(modelFamily);
        String baseRedirectWarning = buildBaseRedirectWarning(modelFamily, configuredBaseDir, activeBaseDir);
        if (baseRedirectWarning != null) {
            Log.w(TAG, "BASE_REDIRECT: " + baseRedirectWarning);
            updateWarningStatus("⚠ ROUTE: " + baseRedirectWarning);
        }

        if (!shouldUseRootShell(activeBaseDir, PYTHON) && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            if (!storageAccessPromptShown) {
                storageAccessPromptShown = true;
                ensureExternalStorageAccess(activeBaseDir);
            }
            statusText.setText("Нужен доступ ко всем файлам для чтения " + activeBaseDir +
                "\nНажмите Generate или откройте системное разрешение вручную");
            return;
        }
        storageAccessPromptShown = false;
        File ctx = new File(activeBaseDir, "context");
        if (!ctx.exists()) {
            String modeLabel = MODEL_FAMILY_WAN21.equals(modelFamily) ? "WAN 2.1" : "SDXL";
            statusText.setText(modeLabel + " assets не найдены в " + activeBaseDir +
                "\nДеплойте runtime/contexts на телефон" +
                "\nили измените путь в Настройках (⚙)");
            return;
        }
        statusText.setText(MODEL_FAMILY_WAN21.equals(modelFamily)
            ? "Готово к WAN 2.1 runtime probe"
            : getString(R.string.status_idle));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        menu.add(0, 1, 0, R.string.settings)
            .setIcon(android.R.drawable.ic_menu_preferences)
            .setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == 1) {
            Intent intent = new Intent(this, SettingsActivity.class);
            startActivityForResult(intent, 100);
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 100 && resultCode == RESULT_OK) {
            loadSettings();
            refreshSizePresetSpinnerItems();
            selectSizePresetForCurrentInputs();
            checkPrerequisites();
        }
    }

    private void startGeneration() {
        String modelFamily = getSelectedModelFamily();
        String activeBaseDir = resolveActiveBaseDir(modelFamily);
        String configuredBaseDir = resolveConfiguredBaseDir(modelFamily);
        String baseRedirectWarning = buildBaseRedirectWarning(modelFamily, configuredBaseDir, activeBaseDir);
        if (!ensureExternalStorageAccess(activeBaseDir)) {
            return;
        }

        String prompt = promptInput.getText().toString().trim();
        if (MODEL_FAMILY_SDXL.equals(modelFamily) && prompt.isEmpty()) {
            Toast.makeText(this, "Введите промпт", Toast.LENGTH_SHORT).show();
            return;
        }

        String seedStr = seedInput.getText().toString().trim();
        long seed;
        try {
            seed = seedStr.isEmpty()
                ? (new Random().nextInt(900000) + 100000)
                : Long.parseLong(seedStr);
        } catch (NumberFormatException e) {
            Toast.makeText(this, "Seed должен быть числом", Toast.LENGTH_SHORT).show();
            return;
        }
        int steps = stepsSeekBar.getProgress();
        float cfg = cfgSeekBar.getProgress() / 10f;
        String neg = negPromptInput.getText().toString().trim();
        boolean stretch = contrastStretch.isChecked();
        boolean preview = livePreview.isChecked();
        boolean progCfg = progressiveCfg.isChecked();
        isGenerating = true;
        cancelScheduledPrewarmKill();
        if (MODEL_FAMILY_SDXL.equals(modelFamily)) {
            killPrewarmNow("handoff to foreground generation");
        }

        // Build output name
        String outName = "apk_s" + seed;

        // Resolution
        int imgWidth = 1024;
        int imgHeight = 1024;
        try {
            String wStr = widthInput.getText().toString().trim();
            String hStr = heightInput.getText().toString().trim();
            if (!wStr.isEmpty()) imgWidth = Integer.parseInt(wStr);
            if (!hStr.isEmpty()) imgHeight = Integer.parseInt(hStr);
        } catch (NumberFormatException ignored) {}
        // Round to nearest multiple of 8
        imgWidth = Math.max(256, (imgWidth / 8) * 8);
        imgHeight = Math.max(256, (imgHeight / 8) * 8);
        final int finalWidth = imgWidth;
        final int finalHeight = imgHeight;

        clearDisplayedImage(true);
        generateButton.setEnabled(false);
        stopButton.setVisibility(View.VISIBLE);
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setProgress(0);
        saveButton.setVisibility(View.GONE);
        timingText.setVisibility(View.GONE);
        clearErrorState();
        latestTempStatus = "";
        latestWarningStatus = baseRedirectWarning != null ? "⚠ ROUTE: " + baseRedirectWarning : "";
        latestStageStatus = MODEL_FAMILY_WAN21.equals(modelFamily)
            ? "WAN 2.1 runtime probe..."
            : "Запуск...";
        latestProgress = 0;
        renderStatus();

        executor.execute(() -> {
            try {
                runPipeline(prompt, seed, steps, cfg, neg, stretch, preview, progCfg, outName, finalWidth, finalHeight);
            } catch (Exception e) {
                isGenerating = false;
                mainHandler.post(() -> {
                    latestTempStatus = "";
                    showErrorState(getString(R.string.status_error_short), "Ошибка: " + (e.getMessage() != null ? e.getMessage() : "unknown"));
                    renderStatus();
                    generateButton.setEnabled(true);
                    stopButton.setVisibility(View.GONE);
                    progressBar.setVisibility(View.GONE);
                    restoreOrSchedulePrewarm("idle after failed generation start");
                });
            }
        });
    }

    private void stopGeneration() {
        Process p = currentProcess;
        if (p != null) {
            p.destroyForcibly();
            currentProcess = null;
        }
        cancelPreviewPolling();
        isGenerating = false;
        mainHandler.post(() -> {
            clearErrorState();
            latestTempStatus = "";
            latestStageStatus = "Остановлено";
            renderStatus();
            generateButton.setEnabled(true);
            stopButton.setVisibility(View.GONE);
            progressBar.setVisibility(View.GONE);
            restoreOrSchedulePrewarm("idle after cancelled generation");
        });
    }

    private void updateStatus(String status, int progress) {
        latestStageStatus = status;
        if (progress >= 0) {
            latestProgress = progress;
        }
        renderStatus();
    }

    private void updateTempStatus(String tempStatus) {
        latestTempStatus = tempStatus;
        renderStatus();
    }

    private void updateWarningStatus(String warningStatus) {
        latestWarningStatus = warningStatus;
        renderStatus();
    }

    private void renderStatus() {
        final String stage = latestStageStatus;
        final String warning = latestWarningStatus;
        final String temp = latestTempStatus;
        final int progress = latestProgress;
        mainHandler.post(() -> {
            StringBuilder sb = new StringBuilder();
            if (stage != null && !stage.isEmpty()) {
                sb.append(stage);
            }
            if (warning != null && !warning.isEmpty()) {
                if (sb.length() > 0) {
                    sb.append("\n");
                }
                sb.append(warning);
            }
            if (temp != null && !temp.isEmpty()) {
                if (sb.length() > 0) {
                    sb.append("\n");
                }
                sb.append(temp);
            }
            statusText.setText(sb.toString());
            progressBar.setProgress(progress);
        });
    }

    private void clearErrorState() {
        lastErrorDetails = "";
        if (copyErrorButton != null) {
            copyErrorButton.setVisibility(View.GONE);
        }
    }

    private void showErrorState(String statusMessage, String details) {
        latestStageStatus = statusMessage;
        lastErrorDetails = (details != null && !details.trim().isEmpty())
            ? details.trim()
            : statusMessage;
        if (copyErrorButton != null) {
            copyErrorButton.setVisibility(View.VISIBLE);
        }
    }

    private void copyLastErrorToClipboard() {
        String textToCopy = (lastErrorDetails != null && !lastErrorDetails.trim().isEmpty())
            ? lastErrorDetails.trim()
            : statusText.getText().toString();
        ClipboardManager clipboard = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        if (clipboard == null || textToCopy.isEmpty()) {
            Toast.makeText(this, "Нет текста ошибки для копирования", Toast.LENGTH_SHORT).show();
            return;
        }
        clipboard.setPrimaryClip(ClipData.newPlainText("sdxl_error", textToCopy));
        Toast.makeText(this, getString(R.string.error_copied), Toast.LENGTH_SHORT).show();
    }

    private void runPipeline(String prompt, long seed, int steps,
                             float cfg, String neg, boolean stretch,
                             boolean preview, boolean progCfg, String outName,
                             int imgWidth, int imgHeight)
            throws IOException, InterruptedException {
        String modelFamily = getSelectedModelFamily();
        boolean wanMode = MODEL_FAMILY_WAN21.equals(modelFamily);
        String configuredBaseDir = resolveConfiguredBaseDir(modelFamily);
        String activeBaseDir = resolveActiveBaseDir(modelFamily);
        String baseRedirectWarning = buildBaseRedirectWarning(modelFamily, configuredBaseDir, activeBaseDir);
        ExecutionPlan executionPlan = resolveExecutionPlan(activeBaseDir);
        boolean useRootShell = executionPlan.useRootShell;
        String pythonCommand = executionPlan.pythonCommand;
        Log.i(TAG, "runtimePlan: model=" + modelFamily
            + ", configuredBase=" + configuredBaseDir
            + ", activeBase=" + activeBaseDir
            + ", root=" + useRootShell
            + ", python=" + pythonCommand);
        if (baseRedirectWarning != null) {
            Log.w(TAG, "BASE_REDIRECT: " + baseRedirectWarning);
        }
        File bundledRuntimePayloadDir = getBundledRuntimePayloadDirOrNull();
        String generatorScript = resolveGeneratorScriptPath(bundledRuntimePayloadDir);
        if (!useRootShell && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            mainHandler.post(() -> ensureExternalStorageAccess(activeBaseDir));
            throw new IOException("Нет доступа к общей папке Downloads. Выдайте приложению доступ ко всем файлам.");
        }

        File runtimeRoot = ensureDir(new File(getCacheDir(), "sdxl_runtime"));
        File runtimeWorkDir = ensureDir(new File(runtimeRoot, "work"));
        File runtimeOutputDir = ensureDir(new File(runtimeRoot, "outputs"));
        final String runtimeWorkDirPath = runtimeWorkDir.getAbsolutePath();
        final String runtimeOutputDirPath = runtimeOutputDir.getAbsolutePath();
        final String previewPath = new File(runtimeOutputDir, PREVIEW_PNG_NAME).getAbsolutePath();

        // Build shell script (multi-line — no nested-quote issues)
        StringBuilder script = new StringBuilder();
        appendShellEnvironment(script);
        script.append("export MODEL_TO_NPU_MODEL_FAMILY=\"").append(shellEscape(modelFamily)).append("\"\n");
        script.append("export MODEL_TO_NPU_BASE=\"").append(shellEscape(activeBaseDir)).append("\"\n");
        script.append("export SDXL_QNN_BASE=\"").append(shellEscape(activeBaseDir)).append("\"\n");
        script.append("export SDXL_QNN_WORK_DIR=\"").append(shellEscape(runtimeWorkDirPath)).append("\"\n");
        script.append("export SDXL_QNN_OUTPUT_DIR=\"").append(shellEscape(runtimeOutputDirPath)).append("\"\n");
        script.append("export SDXL_QNN_PREVIEW_PNG=\"").append(shellEscape(previewPath)).append("\"\n");
        script.append("export PYTHONDONTWRITEBYTECODE=1\n");
        script.append("export SDXL_QNN_WIDTH=").append(imgWidth).append("\n");
        script.append("export SDXL_QNN_HEIGHT=").append(imgHeight).append("\n");
        script.append("export SDXL_QNN_USE_MMAP=1\n");
        script.append("export SDXL_QNN_LOG_LEVEL=warn\n");
        script.append("export SDXL_SHOW_TEMP=1\n");
        script.append("export SDXL_TEMP_INTERVAL_SEC=1.0\n");
        script.append("export SDXL_QNN_PERF_PROFILE=").append(APK_QNN_PERF_PROFILE).append("\n");
        script.append("export SDXL_QNN_USE_DAEMON=0\n");
        script.append("export SDXL_QNN_SHARED_SERVER=0\n");
        script.append("export SDXL_QNN_ASYNC_PREP=1\n");
        script.append("export SDXL_QNN_PRESTAGE_RUNTIME=1\n");
        script.append("export SDXL_QNN_PREWARM_ALL_CONTEXTS=")
            .append(APK_AGGRESSIVE_CONTEXT_PRIMING_ENABLED ? "1" : "0")
            .append("\n");
        script.append("export SDXL_QNN_PREWARM_PREVIEW=")
            .append(APK_AGGRESSIVE_CONTEXT_PRIMING_ENABLED && preview && !wanMode ? "1" : "0")
            .append("\n");
        script.append("export SDXL_QNN_CLIP_CACHE=1\n");
        script.append("export SDXL_QNN_PREVIEW_PNG_COMPRESS=0\n");
        script.append("export SDXL_QNN_FINAL_PNG_COMPRESS=0\n");
        if (preview && !wanMode) {
            script.append("export SDXL_QNN_PREVIEW_STRIDE=4\n");
        }
        if (wanMode) {
            script.append("export SDXL_QNN_PROFILING_LEVEL=basic\n");
        }
        File accelLib = bundledRuntimePayloadDir != null
            ? new File(bundledRuntimePayloadDir, "phone_gen/lib/libsdxl_runtime_accel.so")
            : new File(activeBaseDir, "phone_gen/lib/libsdxl_runtime_accel.so");
        if (accelLib.isFile()) {
            script.append("export SDXL_QNN_USE_NATIVE_ACCEL=1\n");
            script.append("export SDXL_QNN_ACCEL_LIB=\"")
                .append(shellEscape(accelLib.getAbsolutePath()))
                .append("\"\n");
        }
        boolean bundledQnnConfigReady = appendBundledRuntimeEnvironment(script, bundledRuntimePayloadDir);
        if (bundledRuntimePayloadDir != null) {
            File bundledTaesdOnnx = new File(bundledRuntimePayloadDir, "phone_gen/taesd_decoder.onnx");
            if (bundledTaesdOnnx.isFile()) {
                script.append("export SDXL_QNN_TAESD_ONNX=\"")
                    .append(shellEscape(bundledTaesdOnnx.getAbsolutePath()))
                    .append("\"\n");
            }

            // Prefer HTP context (NPU) over GPU context for TAESD preview
            File bundledTaesdHtpContext = new File(bundledRuntimePayloadDir, "phone_gen/taesd_htp.bin");
            File bundledTaesdGpuContext = new File(bundledRuntimePayloadDir, "phone_gen/taesd_decoder.serialized.bin.bin");
            File taesdContextToUse = bundledTaesdHtpContext.isFile() ? bundledTaesdHtpContext : bundledTaesdGpuContext;
            if (taesdContextToUse.isFile()) {
                script.append("export SDXL_QNN_TAESD_CONTEXT=\"")
                    .append(shellEscape(taesdContextToUse.getAbsolutePath()))
                    .append("\"\n");
            }

            File bundledTaesdModel = new File(bundledRuntimePayloadDir, "phone_gen/lib/libTAESDDecoder.so");
            if (bundledTaesdModel.isFile()) {
                script.append("export SDXL_QNN_TAESD_MODEL=\"")
                    .append(shellEscape(bundledTaesdModel.getAbsolutePath()))
                    .append("\"\n");
            }

            // TAESD preview disabled in v0.4.8-beta:
            // HTP shares server with UNet → +200ms/step overhead; GPU backend (libQnnGpu.so)
            // fails with dlerror(): null in app process (Android linker namespace restriction).
            // Re-enable when dedicated GPU backend is resolved.
            script.append("export SDXL_QNN_TAESD_BACKEND=off\n");
        }
        if (!bundledQnnConfigReady) {
            script.append("if [ -f \"").append(shellEscape(activeBaseDir)).append("/htp_backend_extensions_lightning.json\" ] && [ -f \"")
                .append(shellEscape(activeBaseDir)).append("/lib/libQnnHtpNetRunExtensions.so\" ]; then\n");
            script.append("  export SDXL_QNN_CONFIG_FILE=\"").append(shellEscape(activeBaseDir))
                .append("/htp_backend_extensions_lightning.json\"\n");
            script.append("fi\n");
        }
        script.append(String.format(Locale.US,
            "export LD_LIBRARY_PATH=\"%s/lib:%s/bin:%s/model:$LD_LIBRARY_PATH\"\n",
            shellEscape(activeBaseDir), shellEscape(activeBaseDir), shellEscape(activeBaseDir)));
        script.append(String.format(Locale.US,
            "export ADSP_LIBRARY_PATH=\"%s/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp\"\n", shellEscape(activeBaseDir)));
        script.append("cd \"").append(shellEscape(activeBaseDir)).append("\"\n");

        script.append("exec \"").append(shellEscape(pythonCommand)).append("\" \"").append(shellEscape(generatorScript)).append("\"");
        if (wanMode) {
            script.append(" --model-family wan21 --check-runtime");
            script.append(" --width ").append(imgWidth);
            script.append(" --height ").append(imgHeight);
            script.append(" --probe-perf burst");
        } else {
            script.append(" \"").append(shellEscape(prompt)).append("\"");
            script.append(" --seed ").append(seed);
            script.append(" --steps ").append(steps);
            script.append(" --cfg ").append(String.format(Locale.US, "%.1f", cfg));
            script.append(" --width ").append(imgWidth);
            script.append(" --height ").append(imgHeight);
            script.append(" --name ").append(outName);
            if (cfg > 1.0f) {
                if (!neg.isEmpty()) {
                    script.append(" --neg \"").append(shellEscape(neg)).append("\"");
                }
            }
            if (!stretch) {
                script.append(" --no-stretch");
            }
            if (preview) {
                script.append(" --preview");
            }
            if (progCfg && cfg > 1.0f) {
                script.append(" --prog-cfg");
            }
        }
        script.append(" 2>&1\n");

        updateStatus(
            wanMode
                ? (useRootShell ? "WAN runtime probe (root shell)..." : "WAN runtime probe...")
                : (useRootShell ? "Запуск (root shell)..." : "Запуск..."),
            2
        );

        ProcessBuilder pb;
        if (useRootShell) {
            String su = findAvailableSuOrNull();
            if (su == null) {
                throw new IOException("Root shell requested, but su binary was not found");
            }
            pb = new ProcessBuilder(su, "--mount-master");
        } else {
            pb = new ProcessBuilder("/system/bin/sh");
        }
        pb.redirectErrorStream(true);
        Process process = pb.start();
        currentProcess = process;

        // Write script to shell stdin — avoids quoting problems
        try (OutputStream os = process.getOutputStream()) {
            os.write(script.toString().getBytes("UTF-8"));
            os.flush();
        }

        StringBuilder timingLog = new StringBuilder();
        StringBuilder rawLog = new StringBuilder();
        String savedPath = null;
        String wanReportJson = null;
        int clipDone = 0;

        if (baseRedirectWarning != null) {
            timingLog.append("ROUTE: ").append(baseRedirectWarning).append("\n");
        }

        // Live preview polling: check for preview_current.png every 2 seconds
        final long[] previewLastModified = {0};
        final Runnable previewPoller = new Runnable() {
            @Override
            public void run() {
                if (!isGenerating || activePreviewPoller != this) return;
                File previewFile = new File(previewPath);
                if (previewFile.exists() && previewFile.lastModified() != previewLastModified[0]) {
                    previewLastModified[0] = previewFile.lastModified();
                    decodePreviewBitmapAsync(previewPath, this);
                }
                if (isGenerating && activePreviewPoller == this) {
                    mainHandler.postDelayed(this, 2000);
                }
            }
        };
        isGenerating = true;
        cancelPreviewPolling();
        if (preview && !wanMode) {
            // Delete stale preview from last run
            new File(previewPath).delete();
            activePreviewPoller = previewPoller;
            mainHandler.postDelayed(previewPoller, 2000);
        }

        int exitCode = -1;
        boolean waitedForProcess = false;
        try {
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (currentProcess == null) break; // stopped
                    if (rawLog.length() < 16000) rawLog.append(line).append("\n");

                    if (wanMode) {
                        if (line.startsWith("WAN_REPORT_JSON:")) {
                            wanReportJson = line.substring("WAN_REPORT_JSON:".length()).trim();
                            continue;
                        }
                        if (line.startsWith("[WAN]")) {
                            timingLog.append(line.trim()).append("\n");
                            if (line.startsWith("[WAN] status:")) {
                                updateStatus(line.substring("[WAN] ".length()).trim(), 15);
                            }
                            continue;
                        }
                        if (line.startsWith("WAN_SUMMARY:") || line.startsWith("WAN_STATUS:")) {
                            timingLog.append(line.trim()).append("\n");
                            continue;
                        }
                    }

                    // Parse CLIP progress
                    Matcher m = PAT_CLIP.matcher(line);
                    if (m.find()) {
                        clipDone++;
                        String kind = m.group(1);
                        int msL = Integer.parseInt(m.group(2));
                        int msG = Integer.parseInt(m.group(3));
                        timingLog.append(String.format(Locale.US,
                            "CLIP %s: L=%dms G=%dms\n", kind, msL, msG));
                        updateStatus("CLIP " + kind + "... " + (msL + msG) + "ms",
                            5 + clipDone * 3);
                        continue;
                    }

                    // Parse UNet steps
                    m = PAT_UNET.matcher(line);
                    if (m.find()) {
                        int step = Integer.parseInt(m.group(1));
                        int total = Integer.parseInt(m.group(2));
                        int ms = Integer.parseInt(m.group(3));
                        timingLog.append(String.format(Locale.US,
                            "  UNet %d/%d: %dms\n", step, total, ms));
                        int pct = 10 + step * 75 / total;
                        updateStatus(String.format(Locale.US,
                            "UNet %d/%d — %dms", step, total, ms), pct);
                        continue;
                    }

                    m = PAT_TEMP_LINE.matcher(line);
                    if (m.find()) {
                        Matcher tempMatcher = PAT_TEMP_ITEM.matcher(m.group(1));
                        String cpu = null;
                        String gpu = null;
                        String npu = null;
                        while (tempMatcher.find()) {
                            String label = tempMatcher.group(1);
                            String value = tempMatcher.group(2);
                            switch (label) {
                                case "CPU":
                                    cpu = value;
                                    break;
                                case "GPU":
                                    gpu = value;
                                    break;
                                case "NPU":
                                    npu = value;
                                    break;
                                default:
                                    break;
                            }
                        }
                        if (cpu != null || gpu != null || npu != null) {
                            updateTempStatus(String.format(Locale.US,
                                "CPU %s°C | GPU %s°C | NPU %s°C",
                                cpu != null ? cpu : "—",
                                gpu != null ? gpu : "—",
                                npu != null ? npu : "—"));
                        }
                        continue;
                    }

                    m = PAT_TAESD_WARNING.matcher(line);
                    if (m.find()) {
                        String warning = m.group(1).trim();
                        if (!warning.isEmpty()) {
                            timingLog.append("TAESD: ").append(warning).append("\n");
                            updateWarningStatus("⚠ TAESD: " + warning);
                        }
                        continue;
                    }

                    // Parse preview timing (useful for diagnostics, but do not override progress)
                    m = PAT_PREV.matcher(line);
                    if (m.find()) {
                        int step = Integer.parseInt(m.group(1));
                        int total = Integer.parseInt(m.group(2));
                        int ms = Integer.parseInt(m.group(3));
                        timingLog.append(String.format(Locale.US,
                            "  Preview %d/%d: %dms\n", step, total, ms));
                        continue;
                    }

                    // Parse VAE
                    m = PAT_VAE.matcher(line);
                    if (m.find()) {
                        int ms = Integer.parseInt(m.group(1));
                        timingLog.append(String.format(Locale.US, "VAE: %dms\n", ms));
                        updateStatus("VAE... " + ms + "ms", 90);
                        continue;
                    }

                    // Parse saved path
                    m = PAT_SAVED.matcher(line);
                    if (m.find()) {
                        savedPath = m.group(1).trim();
                        continue;
                    }

                    // Parse total time
                    m = PAT_TOTAL.matcher(line);
                    if (m.find()) {
                        timingLog.append("Total: ").append(m.group(1)).append("s\n");
                        continue;
                    }

                    // UNet total line
                    if (line.contains("UNet total:")) {
                        timingLog.append(line.trim()).append("\n");
                    }
                }
            }
            exitCode = process.waitFor();
            waitedForProcess = true;
        } finally {
            if (!waitedForProcess && process.isAlive()) {
                process.destroyForcibly();
            }
            if (currentProcess == process) {
                currentProcess = null;
            }
            isGenerating = false;
            cancelPreviewPolling();
        }

        if (wanMode) {
            if (exitCode != 0 && wanReportJson == null) {
                throw new IOException(buildRunFailureMessage(exitCode, useRootShell, rawLog.toString()));
            }

            JSONObject wanReport = null;
            if (wanReportJson != null && !wanReportJson.isEmpty()) {
                try {
                    wanReport = new JSONObject(wanReportJson);
                } catch (Exception e) {
                    timingLog.append("WAN report parse failed: ").append(e.getMessage()).append("\n");
                }
            }

            final String finalWanStatus = formatWanStatus(wanReport);
            final String finalWanTiming = formatWanTiming(
                wanReport,
                rawLog.toString(),
                activeBaseDir,
                imgWidth,
                imgHeight,
                timingLog.toString()
            );
            mainHandler.post(() -> {
                clearErrorState();
                latestTempStatus = "";
                latestStageStatus = finalWanStatus;
                clearDisplayedImage(true);
                saveButton.setVisibility(View.GONE);
                stopButton.setVisibility(View.GONE);
                progressBar.setVisibility(View.GONE);
                generateButton.setEnabled(true);
                renderStatus();
                timingText.setText(finalWanTiming);
                timingText.setVisibility(View.VISIBLE);
                schedulePrewarmKill("idle after WAN runtime check");
            });
            return;
        }

        if (exitCode != 0 && savedPath == null) {
            throw new IOException(buildRunFailureMessage(exitCode, useRootShell, rawLog.toString()));
        }

        // Load the generated PNG
        final String finalPath = savedPath != null
            ? savedPath
            : runtimeOutputDirPath + "/" + outName + ".png";

        File pngFile = new File(finalPath);
        if (!pngFile.exists()) {
            throw new IOException("Файл не найден: " + finalPath);
        }

        Bitmap bitmap = decodeBitmapForDisplay(finalPath, false);
        if (bitmap == null) {
            throw new IOException("Не удалось загрузить изображение: " + finalPath);
        }

        final String finalTiming = timingLog.toString();
        mainHandler.post(() -> {
            clearErrorState();
            latestTempStatus = "";
            latestStageStatus = "Готово!";
            showFinalBitmap(bitmap);
            saveButton.setVisibility(View.VISIBLE);
            stopButton.setVisibility(View.GONE);
            progressBar.setVisibility(View.GONE);
            generateButton.setEnabled(true);
            renderStatus();
            timingText.setText(finalTiming);
            timingText.setVisibility(View.VISIBLE);
            restoreOrSchedulePrewarm("idle after generation");
        });
    }

    /** Escape string for use inside double-quoted shell argument */
    private static String shellEscape(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("$", "\\$")
                .replace("`", "\\`");
    }

    private static File ensureDir(File dir) throws IOException {
        if (dir.exists() || dir.mkdirs()) {
            return dir;
        }
        throw new IOException("Не удалось создать каталог: " + dir.getAbsolutePath());
    }

    private boolean ensureExternalStorageAccess(String activeBaseDir) {
        if (shouldUseRootShell(activeBaseDir, PYTHON)) {
            return true;
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            try {
                Intent intent = new Intent(
                    Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:" + getPackageName()));
                startActivity(intent);
            } catch (Exception ignored) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
                startActivity(intent);
            }
            Toast.makeText(this,
                "Разрешите доступ ко всем файлам для работы с папкой Downloads",
                Toast.LENGTH_LONG).show();
            statusText.setText("Ожидается разрешение на доступ ко всем файлам");
            return false;
        }
        storageAccessPromptShown = false;
        return true;
    }

    private boolean shouldUseRootShell(String baseDir, String pythonCommand) {
        // py_runtime path is in app's own filesDir — never needs root
        String pyRuntimePython = RuntimeBootstrap.findBundledPyRuntimePython(this);
        if (pyRuntimePython != null && pyRuntimePython.equals(pythonCommand)) {
            return false;
        }
        return isLegacyBaseDir(baseDir) || looksLikePrivatePythonPath(pythonCommand);
    }

    private void appendShellEnvironment(StringBuilder script) {
        // Prefer py_runtime (no-root, self-contained Python bundle)
        String pyRuntimePython = RuntimeBootstrap.findBundledPyRuntimePython(this);
        if (pyRuntimePython != null) {
            String home = RuntimeBootstrap.getPyRuntimeHome(this);
            String libDir = RuntimeBootstrap.getPyRuntimeLibDir(this);
            String binDir = new File(home, "bin").getAbsolutePath();
            Log.i(TAG, "runtimeEnv: mode=py_runtime, home=" + home + ", libDir=" + libDir);
            script.append("export PYTHONHOME=\"").append(shellEscape(home)).append("\"\n");
            script.append("export LD_LIBRARY_PATH=\"")
                .append(shellEscape(libDir)).append(":$LD_LIBRARY_PATH\"\n");
            script.append("export PATH=\"")
                .append(shellEscape(binDir)).append(":$PATH\"\n");
            return;
        }
        // Fall back to bundled Termux prefix
        File bundledPrefix = RuntimeBootstrap.getBundledPrefixDir(this);
        if (bundledPrefix.isDirectory()) {
            File bundledBin = new File(bundledPrefix, "bin");
            File bundledLib = new File(bundledPrefix, "lib");
            Log.i(TAG, "runtimeEnv: mode=bundled_termux, prefix=" + bundledPrefix.getAbsolutePath());
            script.append("export PREFIX=\"").append(shellEscape(bundledPrefix.getAbsolutePath())).append("\"\n");
            script.append("export HOME=\"").append(shellEscape(new File(bundledPrefix, "home").getAbsolutePath())).append("\"\n");
            script.append("export LD_LIBRARY_PATH=\"")
                .append(shellEscape(bundledLib.getAbsolutePath()))
                .append(":$LD_LIBRARY_PATH\"\n");
            script.append("export PATH=\"")
                .append(shellEscape(bundledBin.getAbsolutePath()))
                .append(":/data/data/com.termux/files/usr/bin:/data/data/com.termux/files/usr/bin/applets:$PATH\"\n");
            return;
        }
        Log.i(TAG, "runtimeEnv: mode=system_sh_fallback");
        script.append("export PATH=/data/data/com.termux/files/usr/bin:/data/data/com.termux/files/usr/bin/applets:$PATH\n");
    }

    private ExecutionPlan resolveExecutionPlan(String activeBaseDir) throws IOException, InterruptedException {
        // Try to extract py_runtime (no-root, self-contained Python bundle)
        try {
            RuntimeBootstrap.ensurePyRuntimeExtracted(this);
        } catch (IOException ignored) {
            // py_runtime extraction failed or not bundled; continue with other options
        }
        // Also try bundled Termux assets (root path)
        try {
            RuntimeBootstrap.ensureBundledAssetsExtracted(this);
        } catch (IOException ignored) {
            // Fall back to external Python discovery if bundle extraction failed.
        }

        String pyRuntimePython = RuntimeBootstrap.findBundledPyRuntimePython(this);
        String bundledPython = RuntimeBootstrap.findBundledPython(this);
        String detectedPython = SettingsActivity.detectDefaultPython(activeBaseDir);
        LinkedHashSet<String> noRootCandidates = new LinkedHashSet<>();
        LinkedHashSet<String> rootCandidates = new LinkedHashSet<>();

        // py_runtime is the top-priority no-root candidate
        addIfPresent(noRootCandidates, pyRuntimePython);
        addIfPresent(noRootCandidates, bundledPython);
        if (!looksLikePrivatePythonPath(PYTHON)) {
            addIfPresent(noRootCandidates, PYTHON);
        }
        addIfPresent(noRootCandidates, detectedPython);
        if (isSimpleCommandName(PYTHON)) {
            addIfPresent(noRootCandidates, "python3");
            addIfPresent(noRootCandidates, "python");
        }

        addIfPresent(rootCandidates, bundledPython);
        addIfPresent(rootCandidates, PYTHON);
        addIfPresent(rootCandidates, detectedPython);
        addIfPresent(rootCandidates, SettingsActivity.LEGACY_PYTHON);
        if (isSimpleCommandName(PYTHON)) {
            addIfPresent(rootCandidates, "python3");
            addIfPresent(rootCandidates, "python");
        }

        boolean preferRoot = shouldUseRootShell(activeBaseDir, PYTHON);
        if (preferRoot) {
            ExecutionPlan rootPlan = firstUsablePlan(rootCandidates, true);
            if (rootPlan != null) {
                return rootPlan;
            }
        }

        ExecutionPlan noRootPlan = firstUsablePlan(noRootCandidates, false);
        if (noRootPlan != null) {
            return noRootPlan;
        }

        if (!preferRoot) {
            ExecutionPlan rootFallback = firstUsablePlan(rootCandidates, true);
            if (rootFallback != null) {
                return rootFallback;
            }
        }

        throw new IOException(
            "Не удалось найти исполняемый Python runtime. Откройте Настройки → Проверить файлы, "
                + "проверьте путь к Python и, при необходимости, извлеките bundled offline runtime."
        );
    }

    private ExecutionPlan firstUsablePlan(Set<String> candidates, boolean useRootShell)
            throws IOException, InterruptedException {
        if (useRootShell && !hasWorkingRootShell()) {
            return null;
        }
        for (String candidate : candidates) {
            if (candidate == null) {
                continue;
            }
            String normalized = candidate.trim();
            if (normalized.isEmpty()) {
                continue;
            }
            if (canExecutePython(useRootShell, normalized)) {
                return new ExecutionPlan(useRootShell, normalized);
            }
        }
        return null;
    }

    private boolean canExecutePython(boolean useRootShell, String pythonCommand)
            throws IOException, InterruptedException {
        StringBuilder script = new StringBuilder();
        appendShellEnvironment(script);
        if (isSimpleCommandName(pythonCommand)) {
            script.append("if command -v \"")
                .append(shellEscape(pythonCommand))
                .append("\" >/dev/null 2>&1; then echo OK; else echo MISS; fi\n");
        } else {
            script.append("if [ -x \"")
                .append(shellEscape(pythonCommand))
                .append("\" ]; then echo OK; else echo MISS; fi\n");
        }
        String output = runShellScriptForOutput(useRootShell, script.toString(), 15);
        return output.contains("OK");
    }

    private boolean hasWorkingRootShell() {
        if (rootShellAvailable != null) {
            return rootShellAvailable;
        }
        try {
            String output = runShellScriptForOutput(true, "id -u\n", 10);
            rootShellAvailable = output.contains("0");
        } catch (Exception e) {
            rootShellAvailable = false;
        }
        return rootShellAvailable;
    }

    private String runShellScriptForOutput(boolean useRootShell, String script, int timeoutSeconds)
            throws IOException, InterruptedException {
        ProcessBuilder pb;
        if (useRootShell) {
            String su = findAvailableSuOrNull();
            if (su == null) {
                throw new IOException("su binary not found");
            }
            pb = new ProcessBuilder(su, "--mount-master");
        } else {
            pb = new ProcessBuilder("/system/bin/sh");
        }
        pb.redirectErrorStream(true);
        Process process = pb.start();
        String scriptToRun = script.endsWith("\n") ? script : script + "\n";
        scriptToRun += "exit\n";
        Log.i(TAG, "shellProbe: start root=" + useRootShell + ", timeout=" + timeoutSeconds + "s");
        try (OutputStream os = process.getOutputStream()) {
            os.write(scriptToRun.getBytes("UTF-8"));
            os.flush();
        }

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append('\n');
            }
        }

        boolean finished = process.waitFor(timeoutSeconds, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            Log.w(TAG, "shellProbe: timeout root=" + useRootShell);
            throw new IOException("shell probe timeout");
        }
        Log.i(TAG, "shellProbe: done root=" + useRootShell + ", exit=" + process.exitValue()
            + ", output=" + output.toString().trim());
        return output.toString();
    }

    private static void addIfPresent(Set<String> target, String value) {
        if (value == null) {
            return;
        }
        String normalized = value.trim();
        if (!normalized.isEmpty()) {
            target.add(normalized);
        }
    }

    private static boolean looksLikePrivatePythonPath(String pythonCommand) {
        if (pythonCommand == null) {
            return false;
        }
        String normalized = pythonCommand.trim();
        for (String prefix : TERMUX_PRIVATE_PREFIXES) {
            if (normalized.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isSimpleCommandName(String command) {
        return command != null
            && !command.trim().isEmpty()
            && !command.contains("/")
            && !command.contains("\\");
    }

    private static String findAvailableSuOrNull() {
        for (String path : new String[]{
            "/product/bin/su",
            "/sbin/su", "/system/xbin/su", "/system/bin/su",
            "/su/bin/su", "/data/adb/magisk/su"
        }) {
            if (new File(path).exists()) {
                return path;
            }
        }
        return null;
    }

    private static final class ExecutionPlan {
        final boolean useRootShell;
        final String pythonCommand;

        ExecutionPlan(boolean useRootShell, String pythonCommand) {
            this.useRootShell = useRootShell;
            this.pythonCommand = pythonCommand;
        }
    }

    private static String findSu() {
        String available = findAvailableSuOrNull();
        return available != null ? available : "su";
    }

    private File getBundledRuntimePayloadDirOrNull() throws IOException {
        RuntimeBootstrap.ensureBundledAssetsExtracted(this);
        File payloadDir = RuntimeBootstrap.getBundledRuntimePayloadDir(this);
        return payloadDir.isDirectory() ? payloadDir : null;
    }

    private String resolveGeneratorScriptPath(File bundledRuntimePayloadDir) {
        if (bundledRuntimePayloadDir != null) {
            File bundledGenerator = new File(bundledRuntimePayloadDir, "phone_gen/generate.py");
            if (bundledGenerator.isFile()) {
                return bundledGenerator.getAbsolutePath();
            }
        }
        return GEN_SCRIPT;
    }

    private String buildRunFailureMessage(int exitCode, boolean useRootShell, String rawLog) {
        String safeRawLog = rawLog != null ? rawLog : "";
        String hint;
        if (
                safeRawLog.contains("Device Creation failure")
                    || safeRawLog.contains("contextCreateFromBinary_failed")
                    || safeRawLog.contains("Failed to load skel")
                    || safeRawLog.contains("QnnDsp <E>")
                    || safeRawLog.contains("qnn-net-run failed: exit 11")
        ) {
            hint = "QNN/HTP runtime не смог поднять backend на телефоне.\n"
                + "Проверьте staged runtime paths, HTP skel/runtime libs и доступность DSP backend.";
        } else if (exitCode == 127) {
            hint = "Команда не найдена (код 127).\nПроверьте путь/команду Python в Настройках или извлеките bundled runtime через Проверку в Настройках.";
        } else if (exitCode == 2 && safeRawLog.contains("unrecognized arguments")) {
            hint = "Python runtime запустился, но phone-side generate.py не понял новые аргументы.\n"
                + "Обычно это означает устаревший runtime на телефоне. Обновите APK до свежей версии: она запускает bundled generate.py и актуальные fast-path бинарники.";
        } else if (useRootShell && (exitCode == 1 || exitCode == 13 || exitCode == 126)) {
            hint = "Root-доступ не предоставлен или окружение Termux недоступно.\nПроверьте Magisk и путь к Python в Настройках.";
        } else if (exitCode == 1 || exitCode == 13 || exitCode == 126) {
            hint = "Нет доступа к файлам или исполняемым компонентам.\nПроверьте путь к Downloads и команду Python в Настройках.";
        } else {
            hint = "Генерация завершилась с ошибкой (код " + exitCode + ")";
        }
        String details = !safeRawLog.trim().isEmpty()
            ? "\n\n" + safeRawLog.trim()
            : "";
        return hint + details;
    }

    private String formatWanStatus(JSONObject report) {
        if (report == null) {
            return "WAN runtime probe завершён";
        }
        String status = report.optString("status", "UNKNOWN");
        return "WAN runtime probe: " + status;
    }

    private void appendWanProbeRun(StringBuilder sb, JSONObject probeRuns, String key, String label) {
        if (probeRuns == null) {
            return;
        }
        JSONObject run = probeRuns.optJSONObject(key);
        if (run == null) {
            return;
        }
        sb.append(label)
            .append(": wall=")
            .append(String.format(Locale.US, "%.1f", run.optDouble("wall_ms", 0.0)))
            .append("ms, reported=")
            .append(String.format(Locale.US, "%.1f", run.optDouble("reported_ms", 0.0)))
            .append("ms, overhead=")
            .append(String.format(Locale.US, "%.1f", run.optDouble("overhead_ms", 0.0)))
            .append("ms\n");
    }

    private void appendJsonArray(StringBuilder sb, String title, JSONArray values) {
        if (values == null || values.length() == 0) {
            return;
        }
        sb.append(title).append(":\n");
        for (int i = 0; i < values.length(); i++) {
            sb.append("- ").append(values.optString(i)).append("\n");
        }
    }

    private String formatWanTiming(
            JSONObject report,
            String rawLog,
            String activeBaseDir,
            int imgWidth,
            int imgHeight,
            String parsedLines) {
        if (report == null) {
            return !parsedLines.trim().isEmpty() ? parsedLines.trim() : rawLog.trim();
        }

        StringBuilder sb = new StringBuilder();
        sb.append("WAN 2.1 runtime probe (не генерация)\n");
        sb.append("Status: ").append(report.optString("status", "UNKNOWN")).append("\n");
        sb.append("Base: ").append(report.optString("base_dir", activeBaseDir)).append("\n");
        sb.append("Requested: ").append(report.optString("requested_resolution", imgWidth + "x" + imgHeight)).append("\n");
        if (!report.optString("run_tag", "").isEmpty()) {
            sb.append("Run tag: ").append(report.optString("run_tag")).append("\n");
        }
        sb.append("Perf: ").append(report.optString("probe_perf_profile", "burst"));
        sb.append(" | Profiling: ").append(report.optString("qnn_profiling_level", "off")).append("\n");

        JSONObject probeRuns = report.optJSONObject("probe_runs");
        appendWanProbeRun(sb, probeRuns, "direct", "Direct");
        appendWanProbeRun(sb, probeRuns, "server_cold", "Server cold");
        appendWanProbeRun(sb, probeRuns, "server_warm", "Server warm");

        if (report.has("reuse_gain_ms")) {
            sb.append("Reuse gain: ")
                .append(String.format(Locale.US, "%.1f", report.optDouble("reuse_gain_ms", 0.0)))
                .append("ms (")
                .append(String.format(Locale.US, "%.2f", report.optDouble("reuse_gain_pct", 0.0)))
                .append("%)\n");
        }

        appendJsonArray(sb, "Warnings", report.optJSONArray("warnings"));
        appendJsonArray(sb, "Missing", report.optJSONArray("missing"));

        if (!parsedLines.trim().isEmpty()) {
            sb.append("\nParsed log:\n").append(parsedLines.trim()).append("\n");
        }
        return sb.toString().trim();
    }

    private void saveImage() {
        Bitmap bitmapToSave;
        synchronized (bitmapLock) {
            bitmapToSave = currentBitmap;
        }
        if (bitmapToSave == null || bitmapToSave.isRecycled()) return;

        ContentValues values = new ContentValues();
        String filename = "sdxl_npu_" + System.currentTimeMillis() + ".png";
        values.put(MediaStore.Images.Media.DISPLAY_NAME, filename);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.RELATIVE_PATH,
            Environment.DIRECTORY_PICTURES + "/SDXL_NPU");

        Uri uri = getContentResolver().insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        if (uri != null) {
            try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                bitmapToSave.compress(Bitmap.CompressFormat.PNG, 100, out);
                Toast.makeText(this, "Сохранено: " + filename, Toast.LENGTH_LONG).show();
            } catch (IOException e) {
                Toast.makeText(this, "Ошибка сохранения: " + e.getMessage(),
                    Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Cancel pending prewarm kill timer
        killPrewarmNow("activity destroy");
        cancelPreviewPolling();
        clearDisplayedImage(true);
        // Kill generation process
        Process p = currentProcess;
        if (p != null) p.destroyForcibly();
        if (previewExecutor != null) {
            previewExecutor.shutdownNow();
            previewExecutor = null;
        }
        executor.shutdownNow();
    }
}
