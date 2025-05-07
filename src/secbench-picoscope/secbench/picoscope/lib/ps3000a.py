###
# Copyright CEA (Commissariat à l'énergie atomique et aux
# énergies alternatives) (2017-2025)
#
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
###

from ctypes import (
    c_char_p,
    c_double,
    c_float,
    c_int16,
    c_int32,
    c_uint32,
    c_uint64,
    c_void_p,
)
from enum import IntEnum

from .common import def_symbol, load_libps


class PicoInfo(IntEnum):
    """
    Information to be requested to `ps3000aGetUnitInfo`.
    """

    DRIVER_VERSION = 0x00000000
    USB_VERSION = 0x00000001
    HARDWARE_VERSION = 0x00000002
    VARIANT_INFO = 0x00000003
    BATCH_AND_SERIAL = 0x00000004
    CAL_DATE = 0x00000005
    KERNEL_VERSION = 0x00000006
    DIGITAL_HARDWARE_VERSION = 0x00000007
    ANALOGUE_HARDWARE_VERSION = 0x00000008
    FIRMWARE_VERSION_1 = 0x00000009
    FIRMWARE_VERSION_2 = 0x0000000A


def _load_ps3000a_lib(name="ps3000a"):
    lib = load_libps(name)

    # fmt: off

    doc = """ PICO_STATUS ps3000aOpenUnit
        (
            int16_t *handle,
            int8_t  *serial
        ); """
    def_symbol(lib, "ps3000aOpenUnit", c_uint32,
                        [c_void_p, c_char_p], doc)

    doc = """ PICO_STATUS ps3000aOpenUnitAsync
        (
            int16_t *status,
            int8_t  *serial
        ); """
    def_symbol(lib, "ps3000aOpenUnitAsync", c_uint32,
                        [c_void_p, c_char_p], doc)

    doc = """ PICO_STATUS ps3000aOpenUnitProgress
        (
            int16_t *handle,
            int16_t *progressPercent,
            int16_t *complete
        ); """
    def_symbol(lib, "ps3000aOpenUnitProgress",
                        c_uint32, [c_void_p, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetUnitInfo
        (
            int16_t    handle,
            int8_t    *string,
            int16_t    stringLength,
            int16_t   *requiredSize,
            PICO_INFO  info
        ); """
    def_symbol(lib, "ps3000aGetUnitInfo", c_uint32,
                        [c_int16, c_void_p, c_int16, c_void_p, c_uint32],
                        doc)

    doc = """ PICO_STATUS ps3000aFlashLed
        (
            int16_t  handle,
            int16_t  start
        ); """
    def_symbol(lib, "ps3000aFlashLed", c_uint32,
                        [c_int16, c_int16], doc)

    doc = """ PICO_STATUS ps3000aCloseUnit
        (
            int16_t  handle
        ); """
    def_symbol(lib, "ps3000aCloseUnit", c_uint32, [c_int16, ],
                        doc)

    doc = """ PICO_STATUS ps3000aMemorySegments
        (
            int16_t   handle,
            uint32_t  nSegments,
            int32_t  *nMaxSamples
        ); """
    def_symbol(lib, "ps3000aMemorySegments", c_uint32,
                        [c_int16, c_uint32, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aSetChannel
        (
            int16_t          handle,
            PS3000a_CHANNEL  channel,
            int16_t          enabled,
            PS3000a_COUPLING type,
            PS3000a_RANGE    range,
            float            analogOffset
        ); """
    def_symbol(lib, "ps3000aSetChannel", c_uint32,
                        [c_int16, c_int32, c_int16, c_int32, c_int32, c_float],
                        doc)

    doc = """ PICO_STATUS ps3000aSetDigitalPort
        (
            int16_t              handle,
            PS3000a_DIGITAL_PORT port,
            int16_t              enabled,
            int16_t              logicLevel
        ); """
    def_symbol(lib, "ps3000aSetDigitalPort", c_uint32,
                        [c_int16, c_int32, c_int16, c_int16], doc)

    doc = """ PICO_STATUS ps3000aSetBandwidthFilter
        (
            int16_t                    handle,
            PS3000A_CHANNEL            channel,
            PS3000A_BANDWIDTH_LIMITER  bandwidth
        ); """
    def_symbol(lib, "ps3000aSetBandwidthFilter",
                        c_uint32, [c_int16, c_int32, c_int32], doc)

    doc = """ PICO_STATUS ps3000aSetNoOfCaptures
        (
            int16_t   handle,
            uint32_t  nCaptures
        ); """
    def_symbol(lib, "ps3000aSetNoOfCaptures", c_uint32,
                        [c_int16, c_uint32], doc)

    doc = """ PICO_STATUS ps3000aGetTimebase
        (
            int16_t   handle,
            uint32_t  timebase,
            int32_t   noSamples,
            int32_t  *timeIntervalNanoseconds,
            int16_t   oversample,
            int32_t  *maxSamples,
            uint32_t  segmentIndex
        ); """
    def_symbol(lib, "ps3000aGetTimebase", c_uint32,
                        [c_int16, c_uint32, c_int32, c_void_p, c_int16,
                         c_void_p, c_uint32], doc)

    doc = """ PICO_STATUS ps3000aGetTimebase2
        (
            int16_t  handle,
            uint32_t timebase,
            int32_t  noSamples,
            float   *timeIntervalNanoseconds,
            int16_t  oversample,
            int32_t *maxSamples,
            uint32_t segmentIndex
        ); """
    def_symbol(lib, "ps3000aGetTimebase2", c_uint32,
                        [c_int16, c_uint32, c_int32, c_void_p, c_int16,
                         c_void_p, c_uint32], doc)

    doc = """ PICO_STATUS ps3000aSetSigGenArbitrary
        (
            int16_t                     handle,
            int32_t                     offsetVoltage,
            uint32_t                    pkToPk,
            uint32_t                    startDeltaPhase,
            uint32_t                    stopDeltaPhase,
            uint32_t                    deltaPhaseIncrement,
            uint32_t                    dwellCount,
            int16_t                    *arbitraryWaveform,
            int32_t                     arbitraryWaveformSize,
            PS3000A_SWEEP_TYPE          sweepType,
            PS3000A_EXTRA_OPERATIONS    operation,
            PS3000A_INDEX_MODE          indexMode,
            uint32_t                    shots,
            uint32_t                    sweeps,
            PS3000A_SIGGEN_TRIG_TYPE    triggerType,
            PS3000A_SIGGEN_TRIG_SOURCE  triggerSource,
            int16_t                     extInThreshold
        ); """
    def_symbol(lib, "ps3000aSetSigGenArbitrary",
                        c_uint32,
                        [c_int16, c_int32, c_uint32, c_uint32, c_uint32,
                         c_uint32, c_uint32, c_void_p, c_int32, c_int32,
                         c_int32,
                         c_int32, c_uint32, c_uint32, c_int32, c_int32,
                         c_int16], doc)

    doc = """ PICO_STATUS ps3000aSetSigGenBuiltIn
        (
            int16_t                     handle,
            int32_t                     offsetVoltage,
            uint32_t                    pkToPk,
            int16_t                     waveType,
            float                       startFrequency,
            float                       stopFrequency,
            float                       increment,
            float                       dwellTime,
            PS3000A_SWEEP_TYPE          sweepType,
            PS3000A_EXTRA_OPERATIONS    operation,
            uint32_t                    shots,
            uint32_t                    sweeps,
            PS3000A_SIGGEN_TRIG_TYPE    triggerType,
            PS3000A_SIGGEN_TRIG_SOURCE  triggerSource,
            int16_t                     extInThreshold
        ); """
    def_symbol(lib, "ps3000aSetSigGenBuiltIn",
                        c_uint32,
                        [c_int16, c_int32, c_uint32, c_int16, c_float, c_float,
                         c_float, c_float, c_int32, c_int32,
                         c_uint32,
                         c_uint32, c_int32, c_int32, c_int16], doc)

    doc = """ PICO_STATUS ps3000aSetSigGenPropertiesArbitrary
        (
            int16_t                     handle,
            uint32_t                    startDeltaPhase,
            uint32_t                    stopDeltaPhase,
            uint32_t                    deltaPhaseIncrement,
            uint32_t                    dwellCount,
            PS3000A_SWEEP_TYPE          sweepType,
            uint32_t                    shots,
            uint32_t                    sweeps,
            PS3000A_SIGGEN_TRIG_TYPE    triggerType,
            PS3000A_SIGGEN_TRIG_SOURCE  triggerSource,
            int16_t                     extInThreshold
        ); """
    def_symbol(lib,
                        "ps3000aSetSigGenPropertiesArbitrary", c_uint32,
                        [c_int16, c_uint32, c_uint32, c_uint32, c_uint32,
                         c_int32, c_uint32, c_uint32, c_int32, c_int32,
                         c_int16], doc)

    doc = """ PICO_STATUS ps3000aSetSigGenPropertiesBuiltIn
        (
            int16_t                     handle,
            double                      startFrequency,
            double                      stopFrequency,
            double                      increment,
            double                      dwellTime,
            PS3000A_SWEEP_TYPE          sweepType,
            uint32_t                    shots,
            uint32_t                    sweeps,
            PS3000A_SIGGEN_TRIG_TYPE    triggerType,
            PS3000A_SIGGEN_TRIG_SOURCE  triggerSource,
            int16_t                     extInThreshold
        ); """
    def_symbol(lib,
                        "ps3000aSetSigGenPropertiesBuiltIn", c_uint32,
                        [c_int16, c_double, c_double, c_double, c_double,
                         c_int32, c_uint32, c_uint32, c_int32, c_int32,
                         c_int16], doc)

    doc = """ PICO_STATUS ps3000aSigGenFrequencyToPhase
        (
            int16_t             handle,
            double              frequency,
            PS3000A_INDEX_MODE  indexMode,
            uint32_t            bufferLength,
            uint32_t           *phase
        ); """
    def_symbol(lib,
                        "ps3000aSigGenFrequencyToPhase", c_uint32,
                        [c_int16, c_double, c_int32, c_uint32, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aSigGenArbitraryMinMaxValues
        (
            int16_t   handle,
            int16_t  *minArbitraryWaveformValue,
            int16_t  *maxArbitraryWaveformValue,
            uint32_t *minArbitraryWaveformSize,
            uint32_t *maxArbitraryWaveformSize
        ); """
    def_symbol(lib,
                        "ps3000aSigGenArbitraryMinMaxValues", c_uint32,
                        [c_int16, c_void_p, c_void_p, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetMaxEtsValues
        (
            int16_t  handle,
            int16_t *etsCycles,
            int16_t *etsInterleave
        ); """
    def_symbol(lib, "ps3000aGetMaxEtsValues", c_uint32,
                        [c_int16, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aSigGenSoftwareControl
        (
            int16_t  handle,
            int16_t  state
        ); """
    def_symbol(lib,
                        "ps3000aSigGenSoftwareControl", c_uint32,
                        [c_int16, c_int16], doc)

    doc = """ PICO_STATUS ps3000aSetEts
        (
            int16_t           handle,
            PS3000A_ETS_MODE  mode,
            int16_t           etsCycles,
            int16_t           etsInterleave,
            int32_t          *sampleTimePicoseconds
        ); """
    def_symbol(lib, "ps3000aSetEts", c_uint32,
                        [c_int16, c_int32, c_int16, c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aSetSimpleTrigger
        (
            int16_t                      handle,
            int16_t                      enable,
            PS3000A_CHANNEL              source,
            int16_t                      threshold,
            PS3000A_THRESHOLD_DIRECTION  direction,
            uint32_t                     delay,
            int16_t                      autoTrigger_ms
        ); """
    def_symbol(lib, "ps3000aSetSimpleTrigger",
                        c_uint32,
                        [c_int16, c_int16, c_int32, c_int16, c_int32, c_uint32,
                         c_int16], doc)

    doc = """ PICO_STATUS ps3000aSetTriggerDigitalPortProperties
        (
            int16_t                             handle,
            PS3000A_DIGITAL_CHANNEL_DIRECTIONS *directions,
            int16_t                             nDirections
        ); """
    def_symbol(lib,
                        "ps3000aSetTriggerDigitalPortProperties", c_uint32,
                        [c_int16, c_void_p, c_int16], doc)

    doc = """ PICO_STATUS ps3000aSetPulseWidthDigitalPortProperties
        (
            int16_t                             handle,
            PS3000A_DIGITAL_CHANNEL_DIRECTIONS *directions,
            int16_t                             nDirections
        ); """
    def_symbol(lib,
                        "ps3000aSetPulseWidthDigitalPortProperties", c_uint32,
                        [c_int16, c_void_p, c_int16], doc)

    doc = """ PICO_STATUS ps3000aSetTriggerChannelProperties
        (
            int16_t                             handle,
            PS3000A_TRIGGER_CHANNEL_PROPERTIES *channelProperties,
            int16_t                             nChannelProperties,
            int16_t                             auxOutputEnable,
            int32_t                             autoTriggerMilliseconds
        ); """
    def_symbol(lib,
                        "ps3000aSetTriggerChannelProperties", c_uint32,
                        [c_int16, c_void_p, c_int16, c_int16, c_int32], doc)

    doc = """ PICO_STATUS ps3000aSetTriggerChannelConditions
        (
            int16_t                     handle,
            PS3000A_TRIGGER_CONDITIONS *conditions,
            int16_t                     nConditions
        ); """
    def_symbol(lib,
                        "ps3000aSetTriggerChannelConditions", c_uint32,
                        [c_int16, c_void_p, c_int16], doc)

    doc = """ PICO_STATUS ps3000aSetTriggerChannelConditionsV2
        (
            int16_t                        handle,
            PS3000A_TRIGGER_CONDITIONS_V2 *conditions,
            int16_t                        nConditions
        ); """
    def_symbol(lib,
                        "ps3000aSetTriggerChannelConditionsV2", c_uint32,
                        [c_int16, c_void_p, c_int16], doc)

    doc = """ PICO_STATUS ps3000aSetTriggerChannelDirections
        (
            int16_t                      handle,
            PS3000A_THRESHOLD_DIRECTION  channelA,
            PS3000A_THRESHOLD_DIRECTION  channelB,
            PS3000A_THRESHOLD_DIRECTION  channelC,
            PS3000A_THRESHOLD_DIRECTION  channelD,
            PS3000A_THRESHOLD_DIRECTION  ext,
            PS3000A_THRESHOLD_DIRECTION  aux
        ); """
    def_symbol(lib,
                        "ps3000aSetTriggerChannelDirections", c_uint32,
                        [c_int16, c_int32, c_int32, c_int32, c_int32, c_int32,
                         c_int32], doc)

    doc = """ PICO_STATUS ps3000aSetTriggerDelay
        (
            int16_t   handle,
            uint32_t  delay
        ); """
    def_symbol(lib, "ps3000aSetTriggerDelay", c_uint32,
                        [c_int16, c_uint32], doc)

    doc = """ PICO_STATUS ps3000aSetPulseWidthQualifier
        (
            int16_t                      handle,
            PS3000A_PWQ_CONDITIONS      *conditions,
            int16_t                      nConditions,
            PS3000A_THRESHOLD_DIRECTION  direction,
            uint32_t                     lower,
            uint32_t                     upper,
            PS3000A_PULSE_WIDTH_TYPE     type
        ); """
    def_symbol(lib,
                        "ps3000aSetPulseWidthQualifier", c_uint32,
                        [c_int16, c_void_p, c_int16, c_int32, c_uint32,
                         c_uint32, c_int32], doc)

    doc = """ PICO_STATUS ps3000aSetPulseWidthQualifierV2
        (
            int16_t                      handle,
            PS3000A_PWQ_CONDITIONS_V2   *conditions,
            int16_t                      nConditions,
            PS3000A_THRESHOLD_DIRECTION  direction,
            uint32_t                     lower,
            uint32_t                     upper,
            PS3000A_PULSE_WIDTH_TYPE     type
        ); """
    def_symbol(lib,
                        "ps3000aSetPulseWidthQualifierV2", c_uint32,
                        [c_int16, c_void_p, c_int16, c_int32, c_uint32,
                         c_uint32, c_int32], doc)

    doc = """ PICO_STATUS ps3000aIsTriggerOrPulseWidthQualifierEnabled
        (
            int16_t  handle,
            int16_t *triggerEnabled,
            int16_t *pulseWidthQualifierEnabled
        ); """
    def_symbol(lib,
                        "ps3000aIsTriggerOrPulseWidthQualifierEnabled",
                        c_uint32,
                        [c_int16, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetTriggerTimeOffset64
        (
            int16_t             handle,
            int64_t            *time,
            PS3000A_TIME_UNITS *timeUnits,
            uint32_t            segmentIndex
        ); """
    def_symbol(lib,
                        "ps3000aGetTriggerTimeOffset64", c_uint32,
                        [c_int16, c_void_p, c_void_p, c_uint32], doc)

    doc = """ PICO_STATUS ps3000aGetValuesTriggerTimeOffsetBulk64
        (
            int16_t             handle,
            int64_t            *times,
            PS3000A_TIME_UNITS *timeUnits,
            uint32_t            fromSegmentIndex,
            uint32_t            toSegmentIndex
        ); """
    def_symbol(lib,
                        "ps3000aGetValuesTriggerTimeOffsetBulk64", c_uint32,
                        [c_int16, c_void_p, c_void_p, c_uint32, c_uint32], doc)

    doc = """ PICO_STATUS ps3000aGetNoOfCaptures
        (
            int16_t   handle,
            uint32_t *nCaptures
        ); """
    def_symbol(lib, "ps3000aGetNoOfCaptures", c_uint32,
                        [c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetNoOfProcessedCaptures
        (
            int16_t   handle,
            uint32_t *nProcessedCaptures
        ); """
    def_symbol(lib,
                        "ps3000aGetNoOfProcessedCaptures", c_uint32,
                        [c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aSetDataBuffer
        (
            int16_t            handle,
            int32_t            channelOrPort,
            int16_t           *buffer,
            int32_t            bufferLth,
            uint32_t           segmentIndex,
            PS3000a_RATIO_MODE mode
        ); """
    def_symbol(lib, "ps3000aSetDataBuffer", c_uint32,
                        [c_int16, c_int32, c_void_p, c_int32, c_uint32,
                         c_int32], doc)

    doc = """ PICO_STATUS ps3000aSetDataBuffers
        (
            int16_t            handle,
            int32_t            channelOrPort,
            int16_t           *bufferMax,
            int16_t           *bufferMin,
            int32_t            bufferLth,
            uint32_t           segmentIndex,
            PS3000a_RATIO_MODE mode
        ); """
    def_symbol(lib, "ps3000aSetDataBuffers", c_uint32,
                        [c_int16, c_int32, c_void_p, c_void_p, c_int32,
                         c_uint32, c_int32], doc)

    doc = """ PICO_STATUS ps3000aSetEtsTimeBuffer
        (
            int16_t    handle,
            int64_t *buffer,
            int32_t     bufferLth
        ); """
    def_symbol(lib, "ps3000aSetEtsTimeBuffer",
                        c_uint32, [c_int16, c_void_p, c_int32], doc)

    doc = """ PICO_STATUS ps3000aIsReady
        (
            int16_t  handle,
            int16_t *ready
        ); """
    def_symbol(lib, "ps3000aIsReady", c_uint32,
                        [c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aRunBlock
        (
            int16_t            handle,
            int32_t            noOfPreTriggerSamples,
            int32_t            noOfPostTriggerSamples,
            uint32_t           timebase,
            int16_t            oversample,
            int32_t           *timeIndisposedMs,
            uint32_t           segmentIndex,
            ps3000aBlockReady  lpReady,
            void              *pParameter
        ); """
    def_symbol(lib, "ps3000aRunBlock", c_uint32,
                        [c_int16, c_int32, c_int32, c_uint32, c_int16, c_void_p,
                         c_uint32, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aRunStreaming
        (
            int16_t             handle,
            uint32_t            *sampleInterval,
            PS3000A_TIME_UNITS  sampleIntervalTimeUnits,
            uint32_t            maxPreTriggerSamples,
            uint32_t            maxPostPreTriggerSamples,
            int16_t             autoStop,
            uint32_t            downSampleRatio,
            PS3000A_RATIO_MODE  downSampleRatioMode,
            uint32_t            overviewBufferSize
        ); """
    def_symbol(lib, "ps3000aRunStreaming", c_uint32,
                        [c_int16, c_void_p, c_int32, c_uint32, c_uint32,
                         c_int16, c_uint32, c_int32, c_uint32], doc)

    doc = """ PICO_STATUS ps3000aGetStreamingLatestValues
        (
            int16_t                handle,
            ps3000aStreamingReady  lpPs3000aReady,
            void                   *pParameter
        ); """
    def_symbol(lib,
                        "ps3000aGetStreamingLatestValues", c_uint32,
                        [c_int16, c_void_p, c_void_p], doc)

    # doc = """ void *ps3000aStreamingReady
    #     (
    #         int16_t   handle,
    #         int32_t   noOfSamples,
    #         uint32_t  startIndex,
    #         int16_t   overflow,
    #         uint32_t  triggerAt,
    #         int16_t   triggered,
    #         int16_t   autoStop,
    #         void     *pParameter
    #     );
    #     define a python function which accepts the correct arguments, and pass it to the constructor of this type.
    #     """

    # ps3000a.StreamingReadyType = C_CALLBACK_FUNCTION_FACTORY(None,
    #                                                          c_int16,
    #                                                          c_int32,
    #                                                          c_uint32,
    #                                                          c_int16,
    #                                                          c_uint32,
    #                                                          c_int16,
    #                                                          c_int16,
    #                                                          c_void_p)
    #
    # ps3000a.StreamingReadyType.__doc__ = doc

    doc = """ PICO_STATUS ps3000aNoOfStreamingValues
        (
            int16_t   handle,
            uint32_t *noOfValues
        ); """
    def_symbol(lib, "ps3000aNoOfStreamingValues",
                        c_uint32, [c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetMaxDownSampleRatio
    (
      int16_t               handle,
      uint32_t       noOfUnaggreatedSamples,
      uint32_t      *maxDownSampleRatio,
      PS3000A_RATIO_MODE  downSampleRatioMode,
      uint32_t      segmentIndex
    ); """
    def_symbol(lib,
                        "ps3000aGetMaxDownSampleRatio", c_uint32,
                        [c_int16, c_uint32, c_void_p, c_int32, c_uint32], doc)

    doc = """ PICO_STATUS ps3000aGetValues
        (
            int16_t             handle,
            uint32_t            startIndex,
            uint32_t           *noOfSamples,
            uint32_t            downSampleRatio,
            PS3000a_RATIO_MODE  downSampleRatioMode,
            uint32_t            segmentIndex,
            int16_t            *overflow
        ); """
    def_symbol(lib, "ps3000aGetValues", c_uint32,
                        [c_int16, c_uint32, c_void_p, c_uint32, c_int32,
                         c_uint32, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetValuesBulk
        (
            int16_t             handle,
            uint32_t           *noOfSamples,
            uint32_t            fromSegmentIndex,
            uint32_t            toSegmentIndex,
            uint32_t            downSampleRatio,
            PS3000A_RATIO_MODE  downSampleRatioMode,
            int16_t            *overflow
        ); """
    def_symbol(lib, "ps3000aGetValuesBulk", c_uint32,
                        [c_int16, c_void_p, c_uint32, c_uint32, c_uint32,
                         c_int32, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetValuesAsync
        (
            int16_t             handle,
            uint32_t            startIndex,
            uint32_t            noOfSamples,
            uint32_t            downSampleRatio,
            PS3000A_RATIO_MODE  downSampleRatioMode,
            uint32_t            segmentIndex,
            void               *lpDataReady,
            void               *pParameter
        ); """
    def_symbol(lib, "ps3000aGetValuesAsync", c_uint32,
                        [c_int16, c_uint32, c_uint32, c_uint32, c_int32,
                         c_uint32, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetValuesOverlapped
        (
            int16_t             handle,
            uint32_t            startIndex,
            uint32_t           *noOfSamples,
            uint32_t            downSampleRatio,
            PS3000A_RATIO_MODE  downSampleRatioMode,
            uint32_t            segmentIndex,
            int16_t            *overflow
        ); """
    def_symbol(lib, "ps3000aGetValuesOverlapped",
                        c_uint32,
                        [c_int16, c_uint32, c_void_p, c_uint32, c_int32,
                         c_uint32, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetValuesOverlappedBulk
        (
            int16_t             handle,
            uint32_t            startIndex,
            uint32_t           *noOfSamples,
            uint32_t            downSampleRatio,
            PS3000A_RATIO_MODE  downSampleRatioMode,
            uint32_t            fromSegmentIndex,
            uint32_t            toSegmentIndex,
            int16_t            *overflow
        ); """
    def_symbol(lib,"ps3000aGetValuesOverlappedBulk", c_uint32,[c_int16, c_uint32, c_void_p, c_uint32, c_int32, c_uint32, c_uint32, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetTriggerInfoBulk
        (
            int16_t               handle,
            PS3000A_TRIGGER_INFO *triggerInfo,
            uint32_t              fromSegmentIndex,
            uint32_t              toSegmentIndex
        ); """
    def_symbol(lib, "ps3000aGetTriggerInfoBulk", c_uint32, [c_int16, c_void_p, c_uint32, c_uint32], doc)

    doc = """ PICO_STATUS ps3000aStop
        (
            int16_t  handle
        ); """
    def_symbol(lib, "ps3000aStop", c_uint32, [c_int16, ], doc)

    doc = """ PICO_STATUS ps3000aHoldOff
        (
            int16_t               handle,
            uint64_t              holdoff,
            PS3000A_HOLDOFF_TYPE  type
        ); """
    def_symbol(lib, "ps3000aHoldOff", c_uint32, [c_int16, c_uint64, c_int32], doc)

    doc = """ PICO_STATUS ps3000aGetChannelInformation
        (
            int16_t               handle,
            PS3000A_CHANNEL_INFO  info,
            int32_t               probe,
            int32_t              *ranges,
            int32_t              *length,
            int32_t               channels
        ); """
    def_symbol(lib,"ps3000aGetChannelInformation", c_uint32,[c_int16, c_int32, c_int32, c_void_p, c_void_p, c_int32], doc)

    doc = """ PICO_STATUS ps3000aEnumerateUnits
        (
            int16_t *count,
            int8_t  *serials,
            int16_t *serialLth
        ); """
    def_symbol(lib, "ps3000aEnumerateUnits", c_uint32, [c_void_p, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aPingUnit
        (
            int16_t  handle
        ); """
    def_symbol(lib, "ps3000aPingUnit", c_uint32, [c_int16, ], doc)

    doc = """ PICO_STATUS ps3000aMaximumValue
        (
            int16_t  handle,
            int16_t *value
        ); """
    def_symbol(lib, "ps3000aMaximumValue", c_uint32,[c_int16, c_void_p], doc)

    doc = """" PICO_STATUS ps3000aMinimumValue
        (
            int16_t  handle,
            int16_t *value
        ); """
    def_symbol(lib, "ps3000aMinimumValue", c_uint32,[c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetAnalogueOffset
        (
            int16_t           handle,
            PS3000A_RANGE     range,
            PS3000A_COUPLING  coupling,
            float            *maximumVoltage,
            float            *minimumVoltage
        ); """
    def_symbol(lib, "ps3000aGetAnalogueOffset", c_uint32, [c_int16, c_int32, c_int32, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aGetMaxSegments
        (
            int16_t   handle,
            uint32_t *maxSegments
        ); """
    def_symbol(lib, "ps3000aGetMaxSegments", c_uint32,[c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps3000aChangePowerSource
        (
            int16_t     handle,
            PICO_STATUS powerState
        ); """
    def_symbol(lib, "ps3000aChangePowerSource", c_uint32, [c_int16, c_uint32], doc)

    doc = """ PICO_STATUS ps3000aCurrentPowerSource
        (
            int16_t handle
        ); """
    def_symbol(lib, "ps3000aCurrentPowerSource", c_uint32, [c_int16, c_uint32], doc)
    # fmt: on
    return lib


_LIB_PS3000 = None


def load_ps3000a():
    global _LIB_PS3000
    if _LIB_PS3000 is None:
        _LIB_PS3000 = _load_ps3000a_lib()
        return _LIB_PS3000
    else:
        return _LIB_PS3000