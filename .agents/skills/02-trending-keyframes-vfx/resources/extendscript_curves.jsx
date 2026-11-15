/**
 * ExtendScript Helper: Smooth Bezier Keyframe Interpolation, Crop Apertures, Punch-Ins & Speed Ramps
 * Enforces Premiere Pro tick math (254016000000 ticks/sec) and stable nodeId lookup.
 */

var PREMIERE_TICKS_PER_SECOND = 254016000000;

function secondsToTicksString(seconds) {
    return String(Math.round(Number(seconds) * PREMIERE_TICKS_PER_SECOND));
}

function resolveTimelineClip(trackIndexOrNodeId, clipIndex) {
    var seq = app.project.activeSequence;
    if (!seq) return null;

    // 1. Prefer stable nodeId lookup across all video tracks if string nodeId is passed
    if (typeof trackIndexOrNodeId === "string") {
        for (var t = 0; t < seq.videoTracks.numTracks; t++) {
            var vTrack = seq.videoTracks[t];
            for (var c = 0; c < vTrack.clips.numItems; c++) {
                var cand = vTrack.clips[c];
                if (cand.nodeId === trackIndexOrNodeId || cand.name === trackIndexOrNodeId) {
                    return cand;
                }
            }
        }
        return null;
    }

    // 2. Fallback to (trackIndex, clipIndex)
    var track = seq.videoTracks[trackIndexOrNodeId];
    if (!track || clipIndex < 0 || clipIndex >= track.clips.numItems) return null;
    return track.clips[clipIndex];
}

function findComponentProperty(clip, componentName, propertyName) {
    if (!clip || !clip.components) return null;
    for (var i = 0; i < clip.components.numItems; i++) {
        var comp = clip.components[i];
        if (componentName && comp.displayName !== componentName && comp.matchName !== componentName) {
            continue;
        }
        for (var j = 0; j < comp.properties.numItems; j++) {
            var prop = comp.properties[j];
            if (prop.displayName === propertyName) {
                return prop;
            }
        }
    }
    return null;
}

function setKeyframeBezier(trackIndexOrNodeId, clipIndex, propertyName) {
    try {
        var clip = resolveTimelineClip(trackIndexOrNodeId, clipIndex);
        if (!clip) return "ERROR: Clip not found on active sequence";

        var prop = findComponentProperty(clip, null, propertyName);
        if (!prop || !prop.isTimeVarying()) {
            return "WARN: Property " + propertyName + " not found or not time-varying";
        }

        // Cache keys once outside the loop (avoids O(N^2) DOM re-queries)
        var keys = prop.getKeys();
        if (!keys || !keys.length) return "WARN: No keyframes on " + propertyName;

        for (var k = 0; k < keys.length; k++) {
            // 5 = kfInterpMode_Bezier in Premiere Pro ExtendScript DOM
            prop.setInterpolationTypeAtKey(keys[k], 5, true);
        }
        return "SUCCESS: Interpolation updated to Bezier (" + keys.length + " keys) for " + propertyName;
    } catch (e) {
        return "ERROR: " + e.toString();
    }
}

function applyPunchInZoom(nodeId, triggerInClipSec, startScale, endScale, rampDurationSec) {
    try {
        var clip = resolveTimelineClip(nodeId, 0);
        if (!clip) return "ERROR: Clip not found: " + nodeId;

        var scaleProp = findComponentProperty(clip, "Motion", "Scale");
        if (!scaleProp) return "ERROR: Motion -> Scale property not found";

        scaleProp.setTimeVarying(true);
        var t0 = clip.inPoint.seconds + Number(triggerInClipSec);
        var t1 = t0 + Number(rampDurationSec || 0.08);

        scaleProp.addKey(t0);
        scaleProp.setValueAtKey(t0, Number(startScale || 100.0), true);
        scaleProp.addKey(t1);
        scaleProp.setValueAtKey(t1, Number(endScale || 115.0), true);

        var keys = scaleProp.getKeys();
        for (var k = 0; k < keys.length; k++) {
            scaleProp.setInterpolationTypeAtKey(keys[k], 5, true);
        }
        return "SUCCESS: Applied Bezier Punch-In (" + startScale + "% -> " + endScale + "%) on " + clip.name;
    } catch (e) {
        return "ERROR: " + e.toString();
    }
}

function applyCropApertureKeyframes(nodeId, keyframesArray) {
    try {
        var clip = resolveTimelineClip(nodeId, 0);
        if (!clip) return "ERROR: Clip not found: " + nodeId;

        var topProp = findComponentProperty(clip, "Crop", "Top");
        var botProp = findComponentProperty(clip, "Crop", "Bottom");
        if (!topProp || !botProp) {
            return "ERROR: Crop effect (Top/Bottom) not found on clip " + clip.name + ". Add Crop effect first.";
        }

        topProp.setTimeVarying(true);
        botProp.setTimeVarying(true);

        for (var i = 0; i < keyframesArray.length; i++) {
            var kf = keyframesArray[i];
            var tSec = clip.inPoint.seconds + Number(kf.time_sec);
            var halfClose = Number(kf.close_pct) * 0.5;
            topProp.addKey(tSec);
            topProp.setValueAtKey(tSec, halfClose, true);
            botProp.addKey(tSec);
            botProp.setValueAtKey(tSec, halfClose, true);
        }
        return "SUCCESS: Applied crop_no_stretch aperture keyframes (" + keyframesArray.length + " points) on " + clip.name;
    } catch (e) {
        return "ERROR: " + e.toString();
    }
}
