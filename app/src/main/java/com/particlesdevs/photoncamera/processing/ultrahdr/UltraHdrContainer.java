package com.particlesdevs.photoncamera.processing.ultrahdr;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * Assembles a spec-compliant Ultra HDR (JPEG/R, ISO 21496-1) file from an SDR
 * baseline JPEG and a gain-map JPEG, both produced in Java:
 *
 * <ul>
 *   <li>APP1 XMP GContainer packet (primary image) referencing the GainMap</li>
 *   <li>APP2 MPF (Multi-Picture Format) pointing at the gain-map secondary image</li>
 *   <li>the gain-map JPEG appended after the primary, with its own XMP metadata</li>
 * </ul>
 *
 * No external dependencies - pure byte manipulation.
 *
 * <p>The MPF {@code MPEntry} offsets follow the CIPA DC-007 / libultrahdr
 * convention: they are measured relative to the MPF base (the byte just after
 * the {@code MPF\0} signature in the APP2 segment), not from the file start.
 * The gain-map entry's offset is therefore {@code gainMapSoidAbs - mpfBaseAbs};
 * measuring it from the primary SOI instead points the decoder at the wrong
 * bytes and makes the file unparseable (the gain map appears as raw garbage).
 */
public final class UltraHdrContainer {

    private static final byte[] XMP_IDENTIFIER =
            "http://ns.adobe.com/xap/1.0/\u0000".getBytes(StandardCharsets.US_ASCII);
    private static final byte[] MPF_IDENTIFIER = {'M', 'P', 'F', 0};

    private UltraHdrContainer() {}

    /**
     * @param sdrJpeg        baseline SDR JPEG bytes (no Ultra HDR metadata yet)
     * @param gainMapJpeg    baseline gain-map JPEG bytes (will get XMP prepended)
     * @param gainMapMin     gain-map min (base-2 log boost)
     * @param gainMapMax     gain-map max (base-2 log boost)
     * @param hdrCapacityMax max content boost (log2), equals gainMapMax
     * @return assembled Ultra HDR JPEG bytes
     */
    public static byte[] encode(byte[] sdrJpeg, byte[] gainMapJpeg,
                                float gainMapMin, float gainMapMax, float hdrCapacityMax) {
        // Gain-map image XMP must be assembled first so the primary GContainer can
        // reference the gain-map image's full (XMP-inclusive) length.
        final byte[] gainMapXmp = buildGainMapXmp(gainMapMin, gainMapMax, hdrCapacityMax);
        final byte[] gainMapApp1 = buildApp1(gainMapXmp);
        final byte[] gainMapJpegXmp = prependApp1(gainMapJpeg, gainMapApp1);

        final byte[] primaryXmp = buildPrimaryXmp(gainMapJpegXmp.length);
        final byte[] xmpApp1 = buildApp1(primaryXmp);

        // Locate insertion point: right before the first non APPn/COM marker.
        int insertPos = 2; // skip SOI
        while (insertPos + 4 <= sdrJpeg.length) {
            if ((sdrJpeg[insertPos] & 0xFF) != 0xFF) break;
            final int marker = sdrJpeg[insertPos + 1] & 0xFF;
            if ((marker >= 0xE0 && marker <= 0xEF) || marker == 0xFE) {
                final int len = ((sdrJpeg[insertPos + 2] & 0xFF) << 8) | (sdrJpeg[insertPos + 3] & 0xFF);
                insertPos += 2 + len;
            } else {
                break;
            }
        }

        // Layout: [sdrJpeg 0..insertPos][xmpApp1][mpfApp2][sdrJpeg insertPos..end][gainMapJpegXmp]
        final int mpfApp2Len = 90; // FF E2 + length(2) + MPF payload(86)
        // The gain-map SOI sits right after the primary image (which ends at its EOI).
        final int gainMapSoidAbs = xmpApp1.length + mpfApp2Len + sdrJpeg.length; // absolute gain-map SOI
        // MPF base = position right after the 'MPF\0' signature (FFE2_start + 8).
        final int mpfBaseAbs = insertPos + xmpApp1.length + 8;
        // MPF primary size must be the primary image size (up to the gain-map SOI), not the whole file.
        final int primaryLen = gainMapSoidAbs;
        // MPF gain-map offset is relative to the MPF base (per CIPA DC-007 / libultrahdr), not absolute.
        final int gainMapOffset = gainMapSoidAbs - mpfBaseAbs;
        final int gainMapLen = gainMapJpegXmp.length;

        final byte[] mpfApp2 = buildMpf(primaryLen, gainMapLen, gainMapOffset);

        final ByteArrayOutputStream out = new ByteArrayOutputStream(primaryLen + gainMapLen);
        out.write(sdrJpeg, 0, insertPos);
        out.write(xmpApp1, 0, xmpApp1.length);
        out.write(mpfApp2, 0, mpfApp2.length);
        out.write(sdrJpeg, insertPos, sdrJpeg.length - insertPos);
        out.write(gainMapJpegXmp, 0, gainMapJpegXmp.length);
        return out.toByteArray();
    }

    // ------------------------------------------------------------------------
    // XMP packets
    // ------------------------------------------------------------------------

    private static byte[] buildPrimaryXmp(int gainMapLen) {
        final String xml = "<?xpacket begin=\"﻿\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n"
                + "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" x:xmptk=\"PhotonCamera\">\n"
                + " <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n"
                + "  <rdf:Description rdf:about=\"\"\n"
                + "    xmlns:Container=\"http://ns.google.com/photos/1.0/container/\"\n"
                + "    xmlns:Item=\"http://ns.google.com/photos/1.0/container/item/\"\n"
                + "    xmlns:hdrgm=\"http://ns.adobe.com/hdr-gain-map/1.0/\"\n"
                + "    hdrgm:Version=\"1.0\">\n"
                + "   <Container:Directory>\n"
                + "    <rdf:Seq>\n"
                + "     <rdf:li rdf:parseType=\"Resource\">\n"
                + "      <Container:Item Item:Semantic=\"Primary\" Item:Mime=\"image/jpeg\"/>\n"
                + "     </rdf:li>\n"
                + "     <rdf:li rdf:parseType=\"Resource\">\n"
                + "      <Container:Item Item:Semantic=\"GainMap\" Item:Mime=\"image/jpeg\" Item:Length=\""
                + gainMapLen + "\"/>\n"
                + "     </rdf:li>\n"
                + "    </rdf:Seq>\n"
                + "   </Container:Directory>\n"
                + "  </rdf:Description>\n"
                + " </rdf:RDF>\n"
                + "</x:xmpmeta>\n"
                + "<?xpacket end=\"w\"?>\n";
        return xml.getBytes(StandardCharsets.UTF_8);
    }

    private static byte[] buildGainMapXmp(float gMin, float gMax, float hdrCap) {
        final String xml = "<?xpacket begin=\"﻿\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n"
                + "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" x:xmptk=\"PhotonCamera\">\n"
                + " <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n"
                + "  <rdf:Description rdf:about=\"\"\n"
                + "    xmlns:hdrgm=\"http://ns.adobe.com/hdr-gain-map/1.0/\"\n"
                + "    hdrgm:Version=\"1.0\"\n"
                + "    hdrgm:GainMapMin=\"" + fmt(gMin) + "\"\n"
                + "    hdrgm:GainMapMax=\"" + fmt(gMax) + "\"\n"
                + "    hdrgm:Gamma=\"1\"\n"
                + "    hdrgm:OffsetSDR=\"0.015625\"\n"
                + "    hdrgm:OffsetHDR=\"0.015625\"\n"
                + "    hdrgm:HDRCapacityMin=\"0\"\n"
                + "    hdrgm:HDRCapacityMax=\"" + fmt(hdrCap) + "\"\n"
                + "    hdrgm:BaseRenditionIsHDR=\"False\"/>\n"
                + " </rdf:RDF>\n"
                + "</x:xmpmeta>\n"
                + "<?xpacket end=\"w\"?>\n";
        return xml.getBytes(StandardCharsets.UTF_8);
    }

    private static String fmt(float v) {
        if (!Float.isFinite(v)) v = 0f;
        // Fixed decimal: %.6g can emit scientific notation ("1.2345e-05"),
        // which some XMP consumers fail to parse.
        return String.format(java.util.Locale.US, "%.6f", v);
    }

    // ------------------------------------------------------------------------
    // APP1 wrapper (XMP)
    // ------------------------------------------------------------------------

    private static byte[] buildApp1(byte[] packet) {
        int total = 2 + XMP_IDENTIFIER.length + packet.length;
        byte[] padded = packet;
        if (total % 2 != 0) {
            padded = new byte[packet.length + 1];
            System.arraycopy(packet, 0, padded, 0, packet.length);
            padded[packet.length] = (byte) ' ';
            total++;
        }
        final ByteBuffer bb = ByteBuffer.allocate(total + 2).order(ByteOrder.BIG_ENDIAN);
        bb.put((byte) 0xFF);
        bb.put((byte) 0xE1);
        bb.putShort((short) (2 + XMP_IDENTIFIER.length + padded.length));
        bb.put(XMP_IDENTIFIER);
        bb.put(padded);
        return bb.array();
    }

    private static byte[] prependApp1(byte[] jpeg, byte[] app1) {
        final ByteBuffer bb = ByteBuffer.allocate(jpeg.length + app1.length).order(ByteOrder.BIG_ENDIAN);
        bb.put(jpeg, 0, 2); // SOI
        bb.put(app1);
        bb.put(jpeg, 2, jpeg.length - 2);
        return bb.array();
    }

    // ------------------------------------------------------------------------
    // APP2 MPF (Multi-Picture Format)
    // ------------------------------------------------------------------------

    private static byte[] buildMpf(int primaryLen, int gainMapLen, int dataOffset) {
        // Fixed layout: MPF\0(4) + II(2) + 42(2) + ifdOffset(4) + IFD(42) + MPEntry(32) = 86.
        final ByteBuffer data = ByteBuffer.allocate(86).order(ByteOrder.LITTLE_ENDIAN);
        data.put(MPF_IDENTIFIER);                 // 4
        data.put((byte) 0x49); data.put((byte) 0x49); // "II"
        data.putShort((short) 0x2A);
        data.putInt(8);                           // IFD offset (from "II")
        // IFD
        data.putShort((short) 3);                 // entry count
        // Entry 1: MPFormatIdentifier (0xB000), ASCII, "0100"
        data.putShort((short) 0xB000);
        data.putShort((short) 0x0002);
        data.putInt(4);
        data.put((byte) '0'); data.put((byte) '1'); data.put((byte) '0'); data.put((byte) '0');
        // Entry 2: NumberOfImages (0xB001), LONG = 2
        data.putShort((short) 0xB001);
        data.putShort((short) 0x0004);
        data.putInt(1);
        data.putInt(2);
        // Entry 3: MPEntry (0xB002), UNDEFINED, 32 bytes, offset = 50 (from "II")
        data.putShort((short) 0xB002);
        data.putShort((short) 0x0007);
        data.putInt(32);
        data.putInt(50);
        // Next IFD offset
        data.putInt(0);
        // MPEntry data (2 x 16 bytes), per libultrahdr / CIPA DC-007.
        // Offsets are relative to the primary SOI (file offset 0).
        // Primary image: JPEG format (0x00000000) | primary type (0x030000).
        data.putInt(0x00030000);
        data.putInt(primaryLen);
        data.putInt(0);            // data offset (file start)
        data.putInt(0);            // dependence (none)
        // Gain map image: JPEG format only (0x00000000).
        data.putInt(0x00000000);
        data.putInt(gainMapLen);
        data.putInt(dataOffset);
        data.putInt(0);            // dependence (none)

        final byte[] d = data.array();
        final ByteBuffer seg = ByteBuffer.allocate(2 + 2 + d.length).order(ByteOrder.BIG_ENDIAN);
        seg.put((byte) 0xFF);
        seg.put((byte) 0xE2);
        seg.putShort((short) (2 + d.length));
        seg.put(d);
        return seg.array();
    }
}
