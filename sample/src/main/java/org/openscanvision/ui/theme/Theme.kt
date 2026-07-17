package org.openscanvision.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

private val LightColorScheme = lightColorScheme(
    primary = NIBGold,
    onPrimary = NIBBrown,
    primaryContainer = NIBGoldLight,
    onPrimaryContainer = NIBBrown,
    secondary = NIBBrown,
    onSecondary = Color.White,
    secondaryContainer = NIBBrownLight,
    onSecondaryContainer = Color.White,
    background = NIBBackgroundLight,
    onBackground = NIBBrown,
    surface = NIBSurfaceLight,
    onSurface = NIBBrown,
    surfaceVariant = NIBGoldLight,
    onSurfaceVariant = NIBBrown,
    error = NIBError,
    onError = Color.White,
    errorContainer = NIBError.copy(alpha = 0.12f),
    onErrorContainer = NIBError,
    outline = Color(0xFFE5D9C9)
)

private val DarkColorScheme = darkColorScheme(
    primary = NIBGold,
    onPrimary = Color.Black,
    primaryContainer = NIBGoldDark,
    onPrimaryContainer = Color.White,
    secondary = NIBGold,
    onSecondary = Color.Black,
    secondaryContainer = NIBBrown,
    onSecondaryContainer = Color.White,
    background = NIBBackgroundDark,
    onBackground = Color.White,
    surface = NIBSurfaceDark,
    onSurface = Color.White,
    surfaceVariant = NIBBrown,
    onSurfaceVariant = Color.White.copy(alpha = 0.7f),
    error = NIBError,
    onError = Color.White,
    errorContainer = NIBError.copy(alpha = 0.2f),
    onErrorContainer = Color.White,
    outline = Color(0xFF4A2A1A)
)

private val AppTypography = androidx.compose.material3.Typography(
    titleLarge = TextStyle(
        fontWeight = FontWeight.Bold,
        fontSize = 22.sp,
        lineHeight = 28.sp,
        letterSpacing = 0.sp
    ),
    titleMedium = TextStyle(
        fontWeight = FontWeight.SemiBold,
        fontSize = 18.sp,
        lineHeight = 24.sp
    ),
    bodyLarge = TextStyle(
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.5.sp
    ),
    bodyMedium = TextStyle(
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        lineHeight = 20.sp
    ),
    labelLarge = TextStyle(
        fontWeight = FontWeight.SemiBold,
        fontSize = 14.sp,
        lineHeight = 20.sp
    )
)

@Composable
fun OpenScanVisionTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme
    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography,
        content = content
    )
}