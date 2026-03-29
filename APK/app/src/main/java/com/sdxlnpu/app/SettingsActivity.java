package com.sdxlnpu.app;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.MenuItem;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * Settings for model paths and phone-side configuration.
 * Allows user to change the base directory where context binaries
 * and phone_generate.py are located on the phone filesystem.
 */
public class SettingsActivity extends AppCompatActivity {

    public static final String PREFS_NAME = "sdxl_npu_prefs";
    public static final String KEY_BASE_DIR = "base_dir";
    public static final String KEY_PYTHON_PATH = "python_path";
    public static final String DEFAULT_BASE_DIR = "/data/local/tmp/sdxl_qnn";
    public static final String DEFAULT_PYTHON = "/data/data/com.termux/files/usr/bin/python3";

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
        baseDirInput.setText(prefs.getString(KEY_BASE_DIR, DEFAULT_BASE_DIR));
        pythonPathInput.setText(prefs.getString(KEY_PYTHON_PATH, DEFAULT_PYTHON));

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

        if (baseDir.isEmpty()) baseDir = DEFAULT_BASE_DIR;
        if (python.isEmpty()) python = DEFAULT_PYTHON;

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
        baseDirInput.setText(DEFAULT_BASE_DIR);
        pythonPathInput.setText(DEFAULT_PYTHON);
        Toast.makeText(this, "Сброшено к значениям по умолчанию", Toast.LENGTH_SHORT).show();
    }

    private void verifySetup() {
        String baseDir = baseDirInput.getText().toString().trim();
        if (baseDir.isEmpty()) baseDir = DEFAULT_BASE_DIR;

        // Basic validation
        if (baseDir.contains(";") || baseDir.contains("&") || baseDir.contains("|")) {
            Toast.makeText(this, "Недопустимые символы в пути", Toast.LENGTH_SHORT).show();
            return;
        }

        // Check via su if the directory and key files exist
        StringBuilder report = new StringBuilder();
        try {
            String suBin = findSu();
            String checkScript =
                "ls -la " + baseDir + "/context/ 2>&1 | head -20\n" +
                "echo '---'\n" +
                "ls -la " + baseDir + "/phone_gen/generate.py 2>&1\n" +
                "echo '---'\n" +
                "ls -la " + baseDir + "/phone_gen/tokenizer/ 2>&1\n" +
                "echo '---'\n" +
                "ls -la " + baseDir + "/lib/libQnnHtp.so 2>&1\n" +
                "echo '---'\n" +
                "ls -la " + baseDir + "/bin/qnn-net-run 2>&1\n";

            ProcessBuilder pb = new ProcessBuilder(suBin, "--mount-master");
            pb.redirectErrorStream(true);
            Process p = pb.start();
            p.getOutputStream().write(checkScript.getBytes("UTF-8"));
            p.getOutputStream().close();

            BufferedReader reader = new BufferedReader(
                new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                report.append(line).append("\n");
            }
            p.waitFor();
        } catch (Exception e) {
            report.append("Ошибка: ").append(e.getMessage());
        }

        // Show results
        new android.app.AlertDialog.Builder(this)
            .setTitle("Проверка")
            .setMessage(report.toString().trim())
            .setPositiveButton("OK", null)
            .show();
    }

    private static String findSu() {
        for (String path : new String[]{
            "/product/bin/su", "/sbin/su", "/system/xbin/su",
            "/system/bin/su", "/su/bin/su", "/data/adb/magisk/su"
        }) {
            if (new java.io.File(path).exists()) return path;
        }
        return "su";
    }
}
