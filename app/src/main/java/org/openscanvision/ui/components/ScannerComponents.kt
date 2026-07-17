package org.openscanvision.ui.screens

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties

// ─── Candidate name mapping ─────────────────────────────────────
private val candidateNames = mapOf(
    1 to "Dr. Sintayehu W.",
    2 to "Amanuel Gebre",
    3 to "Tigist Hailu",
    4 to "Michael Chen",
    5 to "Liil Taye",
    6 to "Abdurahman J.",
    7 to "Selamawit Hailu",
    8 to "Yonas Tekle",
    9 to "Mulugeta W.",
    10 to "Hirut Alemu",
    11 to "Bekele Girma",
    12 to "Eyerusalem K."
)

@Composable
fun PermissionScreen(
    onRequestPermission: () -> Unit,
    brandGold: Color,
    background: Color,
    cardColor: Color,
    textColor: Color
) {
    Box(
        modifier = Modifier.fillMaxSize().background(background),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier.padding(32.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = cardColor),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(
                modifier = Modifier.padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("📷", fontSize = 48.sp)
                Spacer(Modifier.height(16.dp))
                Text("Camera Permission Required", fontWeight = FontWeight.Bold, fontSize = 20.sp, color = textColor)
                Spacer(Modifier.height(8.dp))
                Text("This app needs camera access to scan voting cards.", color = textColor.copy(alpha = 0.7f), textAlign = TextAlign.Center)
                Spacer(Modifier.height(24.dp))
                Button(
                    onClick = onRequestPermission,
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = brandGold)
                ) {
                    Text("Grant Permission", color = Color.White)
                }
            }
        }
    }
}

@Composable
fun ErrorScreen(
    message: String,
    onRetry: () -> Unit,
    brandGold: Color,
    background: Color,
    cardColor: Color,
    textColor: Color
) {
    Box(
        modifier = Modifier.fillMaxSize().background(background),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier.padding(32.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = cardColor),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(
                modifier = Modifier.padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("⚠️", fontSize = 48.sp)
                Spacer(Modifier.height(16.dp))
                Text("Camera Error", fontWeight = FontWeight.Bold, fontSize = 20.sp, color = textColor)
                Spacer(Modifier.height(8.dp))
                Text(message, color = textColor.copy(alpha = 0.7f), textAlign = TextAlign.Center)
                Spacer(Modifier.height(24.dp))
                Button(
                    onClick = onRetry,
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = brandGold)
                ) {
                    Text("Retry", color = Color.White)
                }
            }
        }
    }
}

@Composable
fun BottomHud(
    isTracking: Boolean,
    captureStatus: String,
    metricsLine: String,
    onForceScan: () -> Unit,
    modifier: Modifier = Modifier,
    brandGold: Color,
    textColor: Color
) {
    Column(
        modifier = modifier
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(Color.Transparent, Color.White.copy(alpha = 0.95f))
                )
            )
            .padding(horizontal = 24.dp, vertical = 20.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Surface(
            shape = RoundedCornerShape(24.dp),
            color = if (isTracking) brandGold else Color(0xFFF57F17),
            modifier = Modifier.padding(bottom = 8.dp)
        ) {
            Text(
                text = if (isTracking) "● Card Detected" else "○ Looking for card...",
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp),
                color = Color.White,
                fontWeight = FontWeight.SemiBold,
                fontSize = 14.sp
            )
        }
        Spacer(Modifier.height(4.dp))
        Text(
            text = captureStatus,
            color = textColor,
            fontSize = 13.sp,
            textAlign = TextAlign.Center,
            maxLines = 2,
            modifier = Modifier.padding(horizontal = 8.dp)
        )
        Spacer(Modifier.height(4.dp))
        Text(
            text = metricsLine,
            color = textColor.copy(alpha = 0.6f),
            fontSize = 11.sp,
            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(12.dp))
        Button(
            onClick = onForceScan,
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = brandGold, contentColor = Color.White),
            modifier = Modifier.fillMaxWidth(0.7f)
        ) {
            Text("Force Scan", fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
fun ScanResultDialog(
    omrBitmap: Bitmap,
    templateName: String?,
    confidence: Float,
    filledIndices: List<Int>,
    voterId: String,
    voterName: String,
    qrToken: String?,
    onDismiss: () -> Unit,
    onAccept: () -> Unit,
    brandGold: Color,
    cardColor: Color,
    background: Color,
    textColor: Color
) {
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            shape = RoundedCornerShape(28.dp),
            color = cardColor,
        ) {
            Column(
                modifier = Modifier
                    .verticalScroll(rememberScrollState())
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Scan Result", fontWeight = FontWeight.Bold, fontSize = 22.sp, color = textColor)
                Spacer(Modifier.height(4.dp))
                Text("Template: ${templateName ?: "Unknown"}  •  Confidence: ${"%.0f".format(confidence * 100)}%",
                    color = textColor.copy(alpha = 0.7f), fontSize = 14.sp)
                Spacer(Modifier.height(16.dp))

                if (voterId.isNotEmpty() || voterName.isNotEmpty()) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = background)
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            if (voterId.isNotEmpty()) {
                                Text("Voter ID", color = brandGold, fontSize = 12.sp)
                                Text(voterId, fontWeight = FontWeight.Bold, fontSize = 18.sp, color = textColor)
                            }
                            if (voterName.isNotEmpty()) {
                                Spacer(Modifier.height(8.dp))
                                Text("Name", color = brandGold, fontSize = 12.sp)
                                Text(voterName, fontWeight = FontWeight.Medium, fontSize = 18.sp, color = textColor)
                            }
                        }
                    }
                    Spacer(Modifier.height(16.dp))
                }

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (qrToken != null) background else Color(0xFFFFF3E0)
                    )
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            "QR Token",
                            color = if (qrToken != null) brandGold else Color(0xFFE65100),
                            fontSize = 12.sp
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            text = qrToken ?: "Not detected",
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 16.sp,
                            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                            color = if (qrToken != null) textColor else Color(0xFFE65100)
                        )
                    }
                }
                Spacer(Modifier.height(16.dp))

                Card(
                    shape = RoundedCornerShape(16.dp),
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
                ) {
                    Image(
                        bitmap = omrBitmap.asImageBitmap(),
                        contentDescription = "OMR detection",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp)
                            .clip(RoundedCornerShape(16.dp))
                    )
                }
                Spacer(Modifier.height(16.dp))

                if (filledIndices.isNotEmpty()) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = background)
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text("Marked Candidates", color = brandGold, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                            Spacer(Modifier.height(8.dp))
                            filledIndices.forEach { index ->
                                val name = candidateNames[index] ?: "Unknown"
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = String.format("%02d", index),
                                        color = brandGold,
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 18.sp
                                    )
                                    Text(
                                        text = name,
                                        color = textColor,
                                        fontSize = 18.sp,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                }
                                Spacer(Modifier.height(4.dp))
                            }
                        }
                    }
                    Spacer(Modifier.height(24.dp))
                } else {
                    Text("No bubbles marked", color = textColor.copy(alpha = 0.5f), fontSize = 16.sp)
                    Spacer(Modifier.height(24.dp))
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = textColor)
                    ) {
                        Text("Retake")
                    }
                    Button(
                        onClick = onAccept,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = brandGold, contentColor = Color.White)
                    ) {
                        Text("Accept & Save")
                    }
                }
            }
        }
    }
}