package com.sdxlnpu.app;

import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.provider.Settings;
import android.view.View;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import android.content.Intent;
import android.content.SharedPreferences;
import android.view.Menu;
import android.view.MenuItem;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
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

    private static final String LEGACY_BASE_DIR = SettingsActivity.LEGACY_BASE_DIR;

    private String BASE_DIR;
    private String GEN_SCRIPT;
    private String OUTPUT_DIR;
    private String PYTHON;

    private EditText promptInput;
    private EditText negPromptInput;
    private EditText seedInput;
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
    private ProgressBar progressBar;
    private TextView statusText;
    private TextView timingText;
    private ImageView imagePreview;

    private ExecutorService executor;
    private Handler mainHandler;
    private Bitmap currentBitmap;
    private volatile Process currentProcess;
    private volatile boolean isGenerating = false;
    private static final String PREVIEW_PNG_NAME = "preview_current.png";

    // Patterns for parsing generate.py stdout
    private static final Pattern PAT_CLIP  = Pattern.compile("^\\[CLIP (cond|uncond)\\]\\s+L=(\\d+)ms G=(\\d+)ms\\s*$");
    private static final Pattern PAT_UNET  = Pattern.compile("^\\s*\\[UNet (\\d+)/(\\d+)\\][^\\n]*?\\s(\\d+)ms(?:\\s|$)");
    private static final Pattern PAT_PREV  = Pattern.compile("^\\s*\\[PREVIEW step (\\d+)/(\\d+)\\]\\s+(?:CPU\\s+)?(\\d+)ms\\s*$");
    private static final Pattern PAT_TEMP_LINE = Pattern.compile("^\\s*\\[TEMP\\]\\s+(.+)$");
    private static final Pattern PAT_TEMP_ITEM = Pattern.compile("(CPU|GPU|NPU)=([\\d.]+)°C");
    private static final Pattern PAT_VAE   = Pattern.compile("^\\[VAE\\]\\s+(\\d+)ms\\s*$");
    private static final Pattern PAT_SAVED = Pattern.compile("Saved:\\s+(.+\\.png)");
    private static final Pattern PAT_TOTAL = Pattern.compile("Total:\\s+([\\d.]+)s");
    private volatile String latestStageStatus = "";
    private volatile String latestTempStatus = "";
    private volatile int latestProgress = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        promptInput = findViewById(R.id.promptInput);
        negPromptInput = findViewById(R.id.negPromptInput);
        seedInput = findViewById(R.id.seedInput);
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
        progressBar = findViewById(R.id.progressBar);
        statusText = findViewById(R.id.statusText);
        timingText = findViewById(R.id.timingText);
        imagePreview = findViewById(R.id.imagePreview);

        executor = Executors.newSingleThreadExecutor();
        mainHandler = new Handler(Looper.getMainLooper());

        loadSettings();

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

        // Check prerequisites
        checkPrerequisites();
    }

    private void loadSettings() {
        SharedPreferences prefs = getSharedPreferences(
            SettingsActivity.PREFS_NAME, MODE_PRIVATE);
        String detectedBaseDir = SettingsActivity.detectDefaultBaseDir();
        String detectedPython = SettingsActivity.detectDefaultPython(detectedBaseDir);
        BASE_DIR = prefs.getString(SettingsActivity.KEY_BASE_DIR,
            detectedBaseDir);
        PYTHON = prefs.getString(SettingsActivity.KEY_PYTHON_PATH,
            detectedPython);
        GEN_SCRIPT = BASE_DIR + "/phone_gen/generate.py";
        OUTPUT_DIR = BASE_DIR + "/outputs";
    }

    private void checkPrerequisites() {
        if (!shouldUseRootShell() && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            statusText.setText("Нужен доступ ко всем файлам для чтения " + BASE_DIR +
                "\nНажмите Generate или откройте системное разрешение вручную");
            return;
        }
        File ctx = new File(BASE_DIR, "context");
        if (!ctx.exists()) {
            statusText.setText("Модели не найдены в " + BASE_DIR +
                "\nДеплойте через scripts/deploy_to_phone.py" +
                "\nили измените путь в Настройках (⚙)");
        }
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
            checkPrerequisites();
        }
    }

    private void startGeneration() {
        if (!ensureExternalStorageAccess()) {
            return;
        }

        String prompt = promptInput.getText().toString().trim();
        if (prompt.isEmpty()) {
            Toast.makeText(this, "Введите промпт", Toast.LENGTH_SHORT).show();
            return;
        }

        String seedStr = seedInput.getText().toString().trim();
        long seed = seedStr.isEmpty()
            ? (new Random().nextInt(900000) + 100000)
            : Long.parseLong(seedStr);
        int steps = stepsSeekBar.getProgress();
        float cfg = cfgSeekBar.getProgress() / 10f;
        String neg = negPromptInput.getText().toString().trim();
        boolean stretch = contrastStretch.isChecked();
        boolean preview = livePreview.isChecked();
        boolean progCfg = progressiveCfg.isChecked();

        // Build output name
        String outName = "apk_s" + seed;

        generateButton.setEnabled(false);
        stopButton.setVisibility(View.VISIBLE);
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setProgress(0);
        saveButton.setVisibility(View.GONE);
        timingText.setVisibility(View.GONE);
        latestTempStatus = "";
        latestStageStatus = "Запуск...";
        latestProgress = 0;
        renderStatus();

        executor.execute(() -> {
            try {
                runPipeline(prompt, seed, steps, cfg, neg, stretch, preview, progCfg, outName);
            } catch (Exception e) {
                mainHandler.post(() -> {
                    latestTempStatus = "";
                    latestStageStatus = "Ошибка: " + e.getMessage();
                    renderStatus();
                    generateButton.setEnabled(true);
                    stopButton.setVisibility(View.GONE);
                    progressBar.setVisibility(View.GONE);
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
        isGenerating = false;
        mainHandler.post(() -> {
            latestTempStatus = "";
            latestStageStatus = "Остановлено";
            renderStatus();
            generateButton.setEnabled(true);
            stopButton.setVisibility(View.GONE);
            progressBar.setVisibility(View.GONE);
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

    private void renderStatus() {
        final String stage = latestStageStatus;
        final String temp = latestTempStatus;
        final int progress = latestProgress;
        mainHandler.post(() -> {
            StringBuilder sb = new StringBuilder();
            if (stage != null && !stage.isEmpty()) {
                sb.append(stage);
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

    private void runPipeline(String prompt, long seed, int steps,
                             float cfg, String neg, boolean stretch,
                             boolean preview, boolean progCfg, String outName)
            throws IOException, InterruptedException {
        boolean useRootShell = shouldUseRootShell();
        if (!useRootShell && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
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
        script.append("export PATH=/data/data/com.termux/files/usr/bin:/data/data/com.termux/files/usr/bin/applets:$PATH\n");
        script.append("export SDXL_QNN_BASE=\"").append(shellEscape(BASE_DIR)).append("\"\n");
        script.append("export SDXL_QNN_WORK_DIR=\"").append(shellEscape(runtimeWorkDirPath)).append("\"\n");
        script.append("export SDXL_QNN_OUTPUT_DIR=\"").append(shellEscape(runtimeOutputDirPath)).append("\"\n");
        script.append("export SDXL_QNN_PREVIEW_PNG=\"").append(shellEscape(previewPath)).append("\"\n");
        script.append("export SDXL_QNN_USE_MMAP=1\n");
        script.append("export SDXL_QNN_LOG_LEVEL=warn\n");
        script.append("export SDXL_SHOW_TEMP=1\n");
        script.append("export SDXL_TEMP_INTERVAL_SEC=1.0\n");
        script.append("export SDXL_QNN_PERF_PROFILE=sustained_high_performance\n");
        script.append("if [ -f \"").append(shellEscape(BASE_DIR)).append("/htp_backend_extensions_lightning.json\" ] && [ -f \"")
            .append(shellEscape(BASE_DIR)).append("/lib/libQnnHtpNetRunExtensions.so\" ]; then\n");
        script.append("  export SDXL_QNN_CONFIG_FILE=\"").append(shellEscape(BASE_DIR))
            .append("/htp_backend_extensions_lightning.json\"\n");
        script.append("fi\n");
        script.append(String.format(Locale.US,
            "export LD_LIBRARY_PATH=\"%s/lib:%s/bin:%s/model:$LD_LIBRARY_PATH\"\n",
            shellEscape(BASE_DIR), shellEscape(BASE_DIR), shellEscape(BASE_DIR)));
        script.append(String.format(Locale.US,
            "export ADSP_LIBRARY_PATH=\"%s/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp\"\n", shellEscape(BASE_DIR)));
        script.append("cd \"").append(shellEscape(BASE_DIR)).append("\"\n");

        script.append("\"").append(shellEscape(PYTHON)).append("\" \"").append(shellEscape(GEN_SCRIPT)).append("\"");
        script.append(" \"").append(shellEscape(prompt)).append("\"");
        script.append(" --seed ").append(seed);
        script.append(" --steps ").append(steps);
        script.append(" --name ").append(outName);
        if (cfg > 1.0f) {
            script.append(" --cfg ").append(String.format(Locale.US, "%.1f", cfg));
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
        script.append(" 2>&1\n");

        updateStatus("Запуск...", 2);

        ProcessBuilder pb;
        if (useRootShell) {
            pb = new ProcessBuilder(findSu(), "--mount-master");
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
        int clipDone = 0;

        // Live preview polling: check for preview_current.png every 2 seconds
        final long[] previewLastModified = {0};
        final Runnable previewPoller = new Runnable() {
            @Override
            public void run() {
                if (!isGenerating) return;
                File previewFile = new File(previewPath);
                if (previewFile.exists() && previewFile.lastModified() != previewLastModified[0]) {
                    previewLastModified[0] = previewFile.lastModified();
                    Bitmap bm = BitmapFactory.decodeFile(previewPath);
                    if (bm != null) {
                        mainHandler.post(() -> imagePreview.setImageBitmap(bm));
                    }
                }
                if (isGenerating) {
                    mainHandler.postDelayed(this, 2000);
                }
            }
        };
        isGenerating = true;
        if (preview) {
            // Delete stale preview from last run
            new File(previewPath).delete();
            mainHandler.postDelayed(previewPoller, 2000);
        }

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (currentProcess == null) break; // stopped
                if (rawLog.length() < 16000) rawLog.append(line).append("\n");

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

        int exitCode = process.waitFor();
        currentProcess = null;
        isGenerating = false;  // stop preview poller

        if (exitCode != 0 && savedPath == null) {
            String hint;
            if (exitCode == 127) {
                hint = "Команда не найдена (код 127).\nПроверьте путь/команду Python в Настройках.";
            } else if (useRootShell && (exitCode == 1 || exitCode == 13 || exitCode == 126)) {
                hint = "Root-доступ не предоставлен или окружение Termux недоступно.\nПроверьте Magisk и путь к Python в Настройках.";
            } else if (exitCode == 1 || exitCode == 13 || exitCode == 126) {
                hint = "Нет доступа к файлам или исполняемым компонентам.\nПроверьте путь к Downloads и команду Python в Настройках.";
            } else {
                hint = "Генерация завершилась с ошибкой (код " + exitCode + ")";
            }
            String details = rawLog.length() > 0
                ? "\n\n" + rawLog.toString().trim() : "";
            throw new IOException(hint + details);
        }

        // Load the generated PNG
        final String finalPath = savedPath != null
            ? savedPath
            : runtimeOutputDirPath + "/" + outName + ".png";

        File pngFile = new File(finalPath);
        if (!pngFile.exists()) {
            throw new IOException("Файл не найден: " + finalPath);
        }

        Bitmap bitmap = BitmapFactory.decodeFile(finalPath);
        if (bitmap == null) {
            throw new IOException("Не удалось загрузить изображение: " + finalPath);
        }

        final String finalTiming = timingLog.toString();
        mainHandler.post(() -> {
            latestTempStatus = "";
            latestStageStatus = "Готово!";
            currentBitmap = bitmap;
            imagePreview.setImageBitmap(bitmap);
            saveButton.setVisibility(View.VISIBLE);
            stopButton.setVisibility(View.GONE);
            progressBar.setVisibility(View.GONE);
            generateButton.setEnabled(true);
            renderStatus();
            timingText.setText(finalTiming);
            timingText.setVisibility(View.VISIBLE);
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

    private boolean ensureExternalStorageAccess() {
        if (shouldUseRootShell()) {
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
        return true;
    }

    private boolean shouldUseRootShell() {
        return BASE_DIR != null && BASE_DIR.startsWith(LEGACY_BASE_DIR);
    }

    private static String findSu() {
        for (String path : new String[]{
            "/product/bin/su",
            "/sbin/su", "/system/xbin/su", "/system/bin/su",
            "/su/bin/su", "/data/adb/magisk/su"
        }) {
            if (new File(path).exists()) return path;
        }
        return "su";
    }

    private void saveImage() {
        if (currentBitmap == null) return;

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
                currentBitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
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
        Process p = currentProcess;
        if (p != null) p.destroyForcibly();
        executor.shutdown();
    }
}
