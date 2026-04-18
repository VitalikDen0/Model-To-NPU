package com.sdxlnpu.app;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.view.MenuItem;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;

import java.io.File;

/**
 * Settings for model paths and phone-side configuration.
 * Allows user to change the base directory where context binaries
 * and phone_generate.py are located on the phone filesystem.
 */
public class SettingsActivity extends AppCompatActivity {

    public static final String PREFS_NAME = "sdxl_npu_prefs";
    public static final String KEY_BASE_DIR = "base_dir";
    public static final String KEY_PYTHON_PATH = "python_path";
    public static final String DOWNLOADS_BASE_DIR = "/sdcard/Download/sdxl_qnn";
    public static final String LEGACY_BASE_DIR = "/data/local/tmp/sdxl_qnn";
    public static final String DEFAULT_BASE_DIR = DOWNLOADS_BASE_DIR;
    public static final String DEFAULT_PYTHON = "python3";
    public static final String LEGACY_PYTHON = "/data/data/com.termux/files/usr/bin/python3";

    private TextInputEditText baseDirInput;
    private TextInputEditText pythonPathInput;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);

        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle(R.string.settings_title);
        }

        baseDirInput = findViewById(R.id.baseDirInput);
        pythonPathInput = findViewById(R.id.pythonPathInput);

        MaterialButton saveBtn = findViewById(R.id.saveSettingsButton);
        MaterialButton resetBtn = findViewById(R.id.resetSettingsButton);
        MaterialButton verifyBtn = findViewById(R.id.verifyButton);

        // Load saved preferences
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String detectedBaseDir = detectDefaultBaseDir();
        String detectedPython = detectDefaultPython(detectedBaseDir);
        baseDirInput.setText(prefs.getString(KEY_BASE_DIR, detectedBaseDir));
        pythonPathInput.setText(prefs.getString(KEY_PYTHON_PATH, detectedPython));

        saveBtn.setOnClickListener(v -> saveSettings());
        resetBtn.setOnClickListener(v -> resetSettings());
        verifyBtn.setOnClickListener(v -> verifySetup());
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void saveSettings() {
        String baseDir = baseDirInput.getText().toString().trim();
        String python = pythonPathInput.getText().toString().trim();

        if (baseDir.isEmpty()) baseDir = detectDefaultBaseDir();
        if (python.isEmpty()) python = detectDefaultPython(baseDir);

        // Basic validation — no command injection
        if (baseDir.contains(";") || baseDir.contains("&") || baseDir.contains("|")
            || python.contains(";") || python.contains("&") || python.contains("|")) {
            Toast.makeText(this, "Недопустимые символы в пути", Toast.LENGTH_SHORT).show();
            return;
        }

        SharedPreferences.Editor editor = getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit();
        editor.putString(KEY_BASE_DIR, baseDir);
        editor.putString(KEY_PYTHON_PATH, python);
        editor.apply();

        Toast.makeText(this, "Настройки сохранены", Toast.LENGTH_SHORT).show();
        setResult(RESULT_OK);
        finish();
    }

    private void resetSettings() {
        String detectedBaseDir = detectDefaultBaseDir();
        baseDirInput.setText(detectedBaseDir);
        pythonPathInput.setText(detectDefaultPython(detectedBaseDir));
        Toast.makeText(this, "Сброшено к значениям по умолчанию", Toast.LENGTH_SHORT).show();
    }

    private void verifySetup() {
        String baseDir = baseDirInput.getText().toString().trim();
        if (baseDir.isEmpty()) baseDir = detectDefaultBaseDir();

        // Basic validation
        if (baseDir.contains(";") || baseDir.contains("&") || baseDir.contains("|")) {
            Toast.makeText(this, "Недопустимые символы в пути", Toast.LENGTH_SHORT).show();
            return;
        }

        File base = new File(baseDir);
        File contextDir = new File(base, "context");
        File generator = new File(base, "phone_gen/generate.py");
        File tokenizerDir = new File(base, "phone_gen/tokenizer");
        File qnnLib = new File(base, "lib/libQnnHtp.so");
        File qnnRunner = new File(base, "bin/qnn-net-run");

        StringBuilder report = new StringBuilder();
        report.append("Base dir: ").append(baseDir).append("\n\n");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            report.append("All files access: ")
                .append(Environment.isExternalStorageManager() ? "granted" : "missing")
                .append("\n\n");
        }

        appendCheck(report, "context/", contextDir.isDirectory(), contextDir.getAbsolutePath());
        appendCheck(report, "phone_gen/generate.py", generator.isFile(), generator.getAbsolutePath());
        appendCheck(report, "phone_gen/tokenizer/", tokenizerDir.isDirectory(), tokenizerDir.getAbsolutePath());
        appendCheck(report, "lib/libQnnHtp.so", qnnLib.isFile(), qnnLib.getAbsolutePath());
        appendCheck(report, "bin/qnn-net-run", qnnRunner.isFile(), qnnRunner.getAbsolutePath());

        report.append("Bundle: ").append(RuntimeBootstrap.describeBundledAssets(this)).append("\n");
        try {
            String extractedBundle = RuntimeBootstrap.ensureBundledAssetsExtracted(this);
            if (extractedBundle != null) {
                File bundleDir = new File(extractedBundle);
                File debsDir = new File(bundleDir, "debs");
                File scriptsDir = new File(bundleDir, "scripts");
                report.append("Extracted bundle: ").append(extractedBundle).append("\n");
                appendCheck(report, "bundle/debs/", debsDir.isDirectory(), debsDir.getAbsolutePath());
                appendCheck(report, "bundle/scripts/", scriptsDir.isDirectory(), scriptsDir.getAbsolutePath());
                String bundledPython = RuntimeBootstrap.findBundledPython(this);
                if (bundledPython != null) {
                    appendCheck(report, "bundle/prefix/bin/python3", true, bundledPython);
                }
            }
        } catch (Exception e) {
            report.append("Extracted bundle: FAILED\n    ")
                .append(e.getMessage())
                .append("\n\n");
        }

        // Show results
        new android.app.AlertDialog.Builder(this)
            .setTitle("Проверка")
            .setMessage(report.toString().trim())
            .setPositiveButton("OK", null)
            .show();
    }

    private static void appendCheck(StringBuilder report, String label, boolean ok, String path) {
        report.append(ok ? "OK   " : "MISS ")
            .append(label)
            .append("\n    ")
            .append(path)
            .append("\n\n");
    }

    public static String detectDefaultBaseDir() {
        if (new File(LEGACY_BASE_DIR).exists()) {
            return LEGACY_BASE_DIR;
        }
        if (new File(DOWNLOADS_BASE_DIR).exists()) {
            return DOWNLOADS_BASE_DIR;
        }
        return DEFAULT_BASE_DIR;
    }

    public static String detectDefaultPython(String baseDir) {
        return baseDir != null && baseDir.startsWith(LEGACY_BASE_DIR)
            ? LEGACY_PYTHON
            : DEFAULT_PYTHON;
    }
}
