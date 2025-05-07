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

"""
PS6000 driver only using direct library calls
"""

from ctypes import (
    c_char_p,
    c_double,
    c_float,
    c_int16,
    c_int32,
    c_uint16,
    c_uint32,
    c_void_p,
)
from enum import IntEnum

from .common import def_symbol, load_libps


class PicoInfo(IntEnum):
    """
    Information to be requested to `ps6000GetUnitInfo`.
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
    MAC_ADDRESS = 0x0000000B
    SHADOW_CAL = 0x0000000C
    IPP_VERSION = 0x0000000D


def _load_ps6000_lib(name="ps6000"):
    lib = load_libps(name)

    # fmt: off
    doc = """ PICO_STATUS ps6000OpenUnit
        (
            int16_t *handle,
            int8_t  *serial
        ); """
    def_symbol(lib, "ps6000OpenUnit", c_uint32, [c_void_p, c_char_p], doc)

    doc = """ PICO_STATUS ps6000OpenUnitAsync
        (
            int16_t *status,
            int8_t  *serial
        ); """
    def_symbol(lib, "ps6000OpenUnitAsync", c_uint32, [c_void_p, c_char_p], doc)

    doc = """ PICO_STATUS ps6000OpenUnitProgress
        (
          int16_t *handle,
          int16_t *progressPercent,
          int16_t *complete
        ); """
    def_symbol(lib, "ps6000OpenUnitProgress", c_uint32, [c_void_p, c_void_p, c_void_p],
               doc)

    doc = """ PICO_STATUS ps6000GetUnitInfo
        (
            int16_t    handle,
            int8_t    *string,
            int16_t    stringLength,
            int16_t   *requiredSize,
            PICO_INFO  info
        ); """
    def_symbol(lib, "ps6000GetUnitInfo", c_uint32,
               [c_int16, c_void_p, c_int16, c_void_p, c_uint32], doc)

    doc = """ PICO_STATUS ps6000FlashLed
        (
            int16_t  handle,
            int16_t  start
        ); """
    def_symbol(lib, "ps6000FlashLed", c_uint32, [c_int16, c_int16], doc)

    doc = """ PICO_STATUS ps6000CloseUnit
        (
            int16_t  handle
        ); """
    def_symbol(lib, "ps6000CloseUnit", c_uint32, [c_int16, ], doc)

    doc = """ PICO_STATUS ps6000MemorySegments
        (
            int16_t   handle,
            uint32_t  nSegments,
            uint32_t *nMaxSamples
        ); """
    def_symbol(lib, "ps6000MemorySegments", c_uint32, [c_int16, c_uint32, c_void_p],
               doc)

    doc = """ PICO_STATUS ps6000SetChannel
        (
            int16_t                   handle,
            PS6000_CHANNEL            channel,
            int16_t                   enabled,
            PS6000_COUPLING           type,
            PS6000_RANGE              range,
            float                     analogueOffset,
            PS6000_BANDWIDTH_LIMITER  bandwidth
        ); """
    def_symbol(lib, "ps6000SetChannel", c_uint32,
               [c_int16, c_int32, c_int16, c_int32, c_int32, c_float, c_int32], doc)

    doc = """ PICO_STATUS ps6000GetTimebase
        (
            int16_t   handle,
            uint32_t  timebase,
            uint32_t  noSamples,
            int32_t  *timeIntervalNanoseconds,
            int16_t   oversample,
            uint32_t *maxSamples,
            uint32_t  segmentIndex
        ); """
    def_symbol(lib, "ps6000GetTimebase", c_uint32,
               [c_int16, c_uint32, c_uint32, c_void_p, c_int16, c_void_p, c_uint32],
               doc)

    doc = """ PICO_STATUS ps6000GetTimebase2
        (
            int16_t   handle,
            uint32_t  timebase,
            uint32_t  noSamples,
            float    *timeIntervalNanoseconds,
            int16_t   oversample,
            uint32_t *maxSamples,
            uint32_t  segmentIndex
        ); """
    def_symbol(lib, "ps6000GetTimebase2", c_uint32,
               [c_int16, c_uint32, c_uint32, c_void_p, c_int16, c_void_p, c_uint32],
               doc)

    doc = """ PICO_STATUS ps6000SetSigGenArbitrary
        (
            int16_t                    handle,
            int32_t                    offsetVoltage,
            uint32_t                   pkToPk,
            uint32_t                   startDeltaPhase,
            uint32_t                   stopDeltaPhase,
            uint32_t                   deltaPhaseIncrement,
            uint32_t                   dwellCount,
            int16_t                   *arbitraryWaveform,
            int32_t                    arbitraryWaveformSize,
            PS6000_SWEEP_TYPE          sweepType,
            PS6000_EXTRA_OPERATIONS    operation,
            PS6000_INDEX_MODE          indexMode,
            uint32_t                   shots,
            uint32_t                   sweeps,
            PS6000_SIGGEN_TRIG_TYPE    triggerType,
            PS6000_SIGGEN_TRIG_SOURCE  triggerSource,
            int16_t                    extInThreshold
        ); """
    def_symbol(lib, "ps6000SetSigGenArbitrary", c_uint32,
               [c_int16, c_int32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32,
                c_void_p, c_int32, c_int32, c_int32, c_int32, c_uint32, c_uint32,
                c_int32, c_int32, c_int16], doc)

    doc = """ PICO_STATUS ps6000SetSigGenBuiltIn
        (
            int16_t                    handle,
            int32_t                    offsetVoltage,
            uint32_t                   pkToPk,
            int16_t                    waveType,
            float                      startFrequency,
            float                      stopFrequency,
            float                      increment,
            float                      dwellTime,
            PS6000_SWEEP_TYPE          sweepType,
            PS6000_EXTRA_OPERATIONS    operation,
            uint32_t                   shots,
            uint32_t                   sweeps,
            PS6000_SIGGEN_TRIG_TYPE    triggerType,
            PS6000_SIGGEN_TRIG_SOURCE  triggerSource,
            int16_t                    extInThreshold
        ); """
    def_symbol(lib, "ps6000SetSigGenBuiltIn", c_uint32,
               [c_int16, c_int32, c_uint32, c_int16, c_float, c_float, c_float, c_float,
                c_int32, c_int32, c_uint32, c_uint32, c_int32, c_int32, c_int16], doc)

    doc = """ PICO_STATUS ps6000SetSigGenBuiltInV2
        (
            int16_t                    handle,
            int32_t                    offsetVoltage,
            uint32_t                   pkToPk,
            int16_t                    waveType,
            double                     startFrequency,
            double                     stopFrequency,
            double                     increment,
            double                     dwellTime,
            PS6000_SWEEP_TYPE          sweepType,
            PS6000_EXTRA_OPERATIONS    operation,
            uint32_t                   shots,
            uint32_t                   sweeps,
            PS6000_SIGGEN_TRIG_TYPE    triggerType,
            PS6000_SIGGEN_TRIG_SOURCE  triggerSource,
            int16_t                    extInThreshold
        ); """
    def_symbol(lib, "ps6000SetSigGenBuiltInV2", c_uint32,
               [c_int16, c_int32, c_uint32, c_int16, c_double, c_double, c_double,
                c_double, c_int32, c_int32, c_uint32, c_uint32, c_int32, c_int32,
                c_int16], doc)

    doc = """ PICO_STATUS ps6000SetSigGenPropertiesArbitrary
        (
            int16_t                    handle,
            int32_t                    offsetVoltage,
            uint32_t                   pkToPk,
            uint32_t                   startDeltaPhase,
            uint32_t                   stopDeltaPhase,
            uint32_t                   deltaPhaseIncrement,
            uint32_t                   dwellCount,
            PS6000_SWEEP_TYPE          sweepType,
            uint32_t                   shots,
            uint32_t                   sweeps,
            PS6000_SIGGEN_TRIG_TYPE    triggerType,
            PS6000_SIGGEN_TRIG_SOURCE  triggerSource,
            int16_t                    extInThreshold
        ); """
    def_symbol(lib, "ps6000SetSigGenPropertiesArbitrary", c_uint32,
               [c_int16, c_int32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32,
                c_int32, c_uint32, c_uint32, c_int32, c_int32, c_int16], doc)

    doc = """ PICO_STATUS ps6000SetSigGenPropertiesBuiltIn
        (
            int16_t                    handle,
            int32_t                    offsetVoltage,
            uint32_t                   pkToPk,
            double                     startFrequency,
            double                     stopFrequency,
            double                     increment,
            double                     dwellTime,
            PS6000_SWEEP_TYPE          sweepType,
            uint32_t                   shots,
            uint32_t                   sweeps,
            PS6000_SIGGEN_TRIG_TYPE    triggerType,
            PS6000_SIGGEN_TRIG_SOURCE  triggerSource,
            int16_t                    extInThreshold
        ); """
    def_symbol(lib, "ps6000SetSigGenPropertiesBuiltIn", c_uint32,
               [c_int16, c_int32, c_uint32, c_double, c_double, c_double, c_double,
                c_int32, c_uint32, c_uint32, c_int32, c_int32, c_int16], doc)

    doc = """ PICO_STATUS ps6000SigGenFrequencyToPhase
        (
            int16_t            handle,
            double             frequency,
            PS6000_INDEX_MODE  indexMode,
            uint32_t           bufferLength,
            uint32_t          *phase
        ); """
    def_symbol(lib, "ps6000SigGenFrequencyToPhase", c_uint32,
               [c_int16, c_double, c_int32, c_uint32, c_void_p], doc)

    doc = """ PICO_STATUS ps6000SigGenArbitraryMinMaxValues
        (
            int16_t   handle,
            int16_t  *minArbitraryWaveformValue,
            int16_t  *maxArbitraryWaveformValue,
            uint32_t *minArbitraryWaveformSize,
            uint32_t *maxArbitraryWaveformSize
        ); """
    def_symbol(lib, "ps6000SigGenArbitraryMinMaxValues", c_uint32,
               [c_int16, c_void_p, c_void_p, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps6000SigGenSoftwareControl
        (
            int16_t  handle,
            int16_t  state
        ); """
    def_symbol(lib, "ps6000SigGenSoftwareControl", c_uint32, [c_int16, c_int16], doc)

    doc = """ PICO_STATUS ps6000SetSimpleTrigger
        (
            int16_t                     handle,
            int16_t                     enable,
            PS6000_CHANNEL              source,
            int16_t                     threshold,
            PS6000_THRESHOLD_DIRECTION  direction,
            uint32_t                    delay,
            int16_t                     autoTrigger_ms
        ); """
    def_symbol(lib, "ps6000SetSimpleTrigger", c_uint32,
               [c_int16, c_int16, c_int32, c_int16, c_int32, c_uint32, c_int16], doc)

    doc = """ PICO_STATUS ps6000SetEts
        (
            int16_t          handle,
            PS6000_ETS_MODE  mode,
            int16_t          etsCycles,
            int16_t          etsInterleave,
            int32_t         *sampleTimePicoseconds
        ); """
    def_symbol(lib, "ps6000SetEts", c_uint32,
               [c_int16, c_int32, c_int16, c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps6000SetTriggerChannelProperties
        (
            int16_t                            handle,
            PS6000_TRIGGER_CHANNEL_PROPERTIES *channelProperties,
            int16_t                            nChannelProperties,
            int16_t                            auxOutputEnable,
            int32_t                            autoTriggerMilliseconds
        ); """
    def_symbol(lib, "ps6000SetTriggerChannelProperties", c_uint32,
               [c_int16, c_void_p, c_int16, c_int16, c_int32], doc)

    doc = """ PICO_STATUS ps6000SetTriggerChannelConditions
        (
            int16_t                    handle,
            PS6000_TRIGGER_CONDITIONS *conditions,
            int16_t                    nConditions
        ); """
    def_symbol(lib, "ps6000SetTriggerChannelConditions", c_uint32,
               [c_int16, c_void_p, c_int16], doc)

    doc = """ PICO_STATUS ps6000SetTriggerChannelDirections
        (
            int16_t                       handle,
            PS6000_THRESHOLD_DIRECTION  channelA,
            PS6000_THRESHOLD_DIRECTION  channelB,
            PS6000_THRESHOLD_DIRECTION  channelC,
            PS6000_THRESHOLD_DIRECTION  channelD,
            PS6000_THRESHOLD_DIRECTION  ext,
            PS6000_THRESHOLD_DIRECTION  aux
        ); """
    def_symbol(lib, "ps6000SetTriggerChannelDirections", c_uint32,
               [c_int16, c_int32, c_int32, c_int32, c_int32, c_int32, c_int32], doc)

    doc = """ PICO_STATUS ps6000SetTriggerDelay
        (
            int16_t   handle,
            uint32_t  delay
        ); """
    def_symbol(lib, "ps6000SetTriggerDelay", c_uint32, [c_int16, c_uint32], doc)

    doc = """ PICO_STATUS ps6000SetPulseWidthQualifier
        (
            int16_t                     handle,
            PS6000_PWQ_CONDITIONS      *conditions,
            int16_t                     nConditions,
            PS6000_THRESHOLD_DIRECTION  direction,
            uint32_t                    lower,
            uint32_t                    upper,
            PS6000_PULSE_WIDTH_TYPE     type
        ); """
    def_symbol(lib, "ps6000SetPulseWidthQualifier", c_uint32,
               [c_int16, c_void_p, c_int16, c_int32, c_uint32, c_uint32, c_int32], doc)

    doc = """ PICO_STATUS ps6000IsTriggerOrPulseWidthQualifierEnabled
        (
            int16_t  handle,
            int16_t *triggerEnabled,
            int16_t *pulseWidthQualifierEnabled
        ); """
    def_symbol(lib, "ps6000IsTriggerOrPulseWidthQualifierEnabled", c_uint32,
               [c_int16, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps6000GetTriggerTimeOffset
        (
            int16_t            handle,
            uint32_t          *timeUpper,
            uint32_t          *timeLower,
            PS6000_TIME_UNITS *timeUnits,
            uint32_t           segmentIndex
        ); """
    def_symbol(lib, "ps6000GetTriggerTimeOffset", c_uint32,
               [c_int16, c_void_p, c_void_p, c_void_p, c_uint32], doc)

    doc = """ PICO_STATUS ps6000GetTriggerTimeOffset64
        (
            int16_t              handle,
            int64_t           *time,
            PS6000_TIME_UNITS *timeUnits,
            uint32_t      segmentIndex
        ); """
    def_symbol(lib, "ps6000GetTriggerTimeOffset64", c_uint32,
               [c_int16, c_void_p, c_void_p, c_uint32], doc)

    doc = """ PICO_STATUS ps6000GetValuesTriggerTimeOffsetBulk
        (
            int16_t            handle,
            uint32_t          *timesUpper,
            uint32_t          *timesLower,
            PS6000_TIME_UNITS *timeUnits,
            uint32_t           fromSegmentIndex,
            uint32_t           toSegmentIndex
        ); """
    def_symbol(lib, "ps6000GetValuesTriggerTimeOffsetBulk", c_uint32,
               [c_int16, c_void_p, c_void_p, c_void_p, c_uint32, c_uint32], doc)

    doc = """ PICO_STATUS ps6000GetValuesTriggerTimeOffsetBulk64
        (
            int16_t            handle,
            int64_t           *times,
            PS6000_TIME_UNITS *timeUnits,
            uint32_t           fromSegmentIndex,
            uint32_t           toSegmentIndex
        ); """
    def_symbol(lib, "ps6000GetValuesTriggerTimeOffsetBulk64", c_uint32,
               [c_int16, c_void_p, c_void_p, c_uint32, c_uint32], doc)

    doc = """ PICO_STATUS ps6000SetDataBuffers
        (
            int16_t            handle,
            PS6000_CHANNEL     channel,
            int16_t           *bufferMax,
            int16_t           *bufferMin,
            uint32_t           bufferLth,
            PS6000_RATIO_MODE  downSampleRatioMode
        ); """
    def_symbol(lib, "ps6000SetDataBuffers", c_uint32,
               [c_int16, c_int32, c_void_p, c_void_p, c_uint32, c_int32], doc)

    doc = """ PICO_STATUS ps6000SetDataBuffer
        (
            int16_t            handle,
            PS6000_CHANNEL     channel,
            int16_t           *buffer,
            uint32_t           bufferLth,
            PS6000_RATIO_MODE  downSampleRatioMode
        ); """
    def_symbol(lib, "ps6000SetDataBuffer", c_uint32,
               [c_int16, c_int32, c_void_p, c_uint32, c_int32], doc)

    doc = """ PICO_STATUS (ps6000SetDataBufferBulk)
        (
            int16_t            handle,
            PS6000_CHANNEL     channel,
            int16_t           *buffer,
            uint32_t           bufferLth,
            uint32_t           waveform,
            PS6000_RATIO_MODE  downSampleRatioMode
        ); """
    def_symbol(lib, "ps6000SetDataBufferBulk", c_uint32,
               [c_int16, c_int32, c_void_p, c_uint32, c_uint32, c_int32], doc)

    doc = """ PICO_STATUS ps6000SetDataBuffersBulk
        (
            int16_t            handle,
            PS6000_CHANNEL     channel,
            int16_t           *bufferMax,
            int16_t           *bufferMin,
            uint32_t           bufferLth,
            uint32_t           waveform,
            PS6000_RATIO_MODE  downSampleRatioMode
        ); """
    def_symbol(lib, "ps6000SetDataBuffersBulk", c_uint32,
               [c_int16, c_int32, c_void_p, c_void_p, c_uint32, c_uint32, c_int32], doc)

    doc = """ PICO_STATUS ps6000SetEtsTimeBuffer
        (
            int16_t   handle,
            int64_t  *buffer,
            uint32_t  bufferLth
        ); """
    def_symbol(lib, "ps6000SetEtsTimeBuffer", c_uint32, [c_int16, c_void_p, c_uint32],
               doc)

    doc = """ PICO_STATUS ps6000SetEtsTimeBuffers
        (
            int16_t   handle,
            uint32_t *timeUpper,
            uint32_t *timeLower,
            uint32_t  bufferLth
        ); """
    def_symbol(lib, "ps6000SetEtsTimeBuffers", c_uint32,
               [c_int16, c_void_p, c_void_p, c_uint32], doc)

    doc = """ PICO_STATUS ps6000RunBlock
        (
            int16_t           handle,
            uint32_t          noOfPreTriggerSamples,
            uint32_t          noOfPostTriggerSamples,
            uint32_t          timebase,
            int16_t           oversample,
            int32_t          *timeIndisposedMs,
            uint32_t          segmentIndex,
            ps6000BlockReady  lpReady,
            void             *pParameter
        ); """
    def_symbol(lib, "ps6000RunBlock", c_uint32,
               [c_int16, c_uint32, c_uint32, c_uint32, c_int16, c_void_p, c_uint32,
                c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps6000IsReady
        (
            int16_t  handle,
            int16_t *ready
        ); """
    def_symbol(lib, "ps6000IsReady", c_uint32, [c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps6000RunStreaming
        (
            int16_t            handle,
            uint32_t          *sampleInterval,
            PS6000_TIME_UNITS  sampleIntervalTimeUnits,
            uint32_t           maxPreTriggerSamples,
            uint32_t           maxPostPreTriggerSamples,
            int16_t            autoStop,
            uint32_t           downSampleRatio,
            PS6000_RATIO_MODE  downSampleRatioMode,
            uint32_t           overviewBufferSize
        ); """
    def_symbol(lib, "ps6000RunStreaming", c_uint32,
               [c_int16, c_void_p, c_int32, c_uint32, c_uint32, c_int16, c_uint32,
                c_int32, c_uint32], doc)

    # FIXME: callbacks not implemented
    # doc = """ void ps6000BlockReady
    #     (
    #         int16_t          handle,
    #         PICO_STATUS      status,
    #         void             *pParameter
    #     ); """
    # ps6000.BlockReadyType = C_CALLBACK_FUNCTION_FACTORY(None,
    #                                                     c_int16,
    #                                                     c_uint32,
    #                                                     c_void_p)
    # ps6000.BlockReadyType.__doc__ = doc

    # FIXME: callbacks not implemented
    # doc = """ void ps6000StreamingReady
    #     (
    #             int16_t                         handle,
    #             uint32_t                        noOfSamples,
    #             uint32_t                        startIndex,
    #             int16_t                         overflow,
    #             uint32_t                        triggerAt,
    #             int16_t                         triggered,
    #             int16_t                         autoStop,
    #             void                            *pParameter
    #     ); """
    # ps6000.StreamingReadyType = C_CALLBACK_FUNCTION_FACTORY(None,
    #                                                         c_int16,
    #                                                         c_uint32,
    #                                                         c_uint32,
    #                                                         c_int16,
    #                                                         c_uint32,
    #                                                         c_int16,
    #                                                         c_int16,
    #                                                         c_void_p)

    # ps6000.StreamingReadyType.__doc__ = doc

    doc = """ PICO_STATUS ps6000GetStreamingLatestValues
        (
            int16_t               handle,
            ps6000StreamingReady  lpPs6000Ready,
            void                 *pParameter
        ); """
    def_symbol(lib, "ps6000GetStreamingLatestValues", c_uint32,
               [c_int16, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps6000NoOfStreamingValues
        (
            int16_t   handle,
            uint32_t *noOfValues
        ); """
    def_symbol(lib, "ps6000NoOfStreamingValues", c_uint32, [c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps6000GetMaxDownSampleRatio
        (
            int16_t            handle,
            uint32_t           noOfUnaggreatedSamples,
            uint32_t          *maxDownSampleRatio,
            PS6000_RATIO_MODE  downSampleRatioMode,
            uint32_t           segmentIndex
        ); """
    def_symbol(lib, "ps6000GetMaxDownSampleRatio", c_uint32,
               [c_int16, c_uint32, c_void_p, c_int32, c_uint32], doc)

    doc = """ PICO_STATUS ps6000GetValues
        (
            int16_t            handle,
            uint32_t           startIndex,
            uint32_t          *noOfSamples,
            uint32_t           downSampleRatio,
            PS6000_RATIO_MODE  downSampleRatioMode,
            uint32_t           segmentIndex,
            int16_t           *overflow
        ); """
    def_symbol(lib, "ps6000GetValues", c_uint32,
               [c_int16, c_uint32, c_void_p, c_uint32, c_int32, c_uint32, c_void_p],
               doc)

    doc = """ PICO_STATUS ps6000GetValuesBulk
        (
            int16_t            handle,
            uint32_t          *noOfSamples,
            uint32_t           fromSegmentIndex,
            uint32_t           toSegmentIndex,
            uint32_t           downSampleRatio,
            PS6000_RATIO_MODE  downSampleRatioMode,
            int16_t           *overflow
        ); """
    def_symbol(lib, "ps6000GetValuesBulk", c_uint32,
               [c_int16, c_void_p, c_uint32, c_uint32, c_uint32, c_int32, c_void_p],
               doc)

    doc = """ PICO_STATUS ps6000GetValuesAsync
        (
            int16_t            handle,
            uint32_t           startIndex,
            uint32_t           noOfSamples,
            uint32_t           downSampleRatio,
            PS6000_RATIO_MODE  downSampleRatioMode,
            uint32_t           segmentIndex,
            void              *lpDataReady,
            void              *pParameter
        ); """
    def_symbol(lib, "ps6000GetValuesAsync", c_uint32,
               [c_int16, c_uint32, c_uint32, c_uint32, c_int32, c_uint32, c_void_p,
                c_void_p], doc)

    doc = """ PICO_STATUS ps6000GetValuesOverlapped
        (
            int16_t            handle,
            uint32_t           startIndex,
            uint32_t          *noOfSamples,
            uint32_t           downSampleRatio,
            PS6000_RATIO_MODE  downSampleRatioMode,
            uint32_t           segmentIndex,
            int16_t           *overflow
        ); """
    def_symbol(lib, "ps6000GetValuesOverlapped", c_uint32,
               [c_int16, c_uint32, c_void_p, c_uint32, c_int32, c_uint32, c_void_p],
               doc)

    doc = """ PICO_STATUS ps6000GetValuesOverlappedBulk
        (
            int16_t            handle,
            uint32_t           startIndex,
            uint32_t          *noOfSamples,
            uint32_t           downSampleRatio,
            PS6000_RATIO_MODE  downSampleRatioMode,
            uint32_t           fromSegmentIndex,
            uint32_t           toSegmentIndex,
            int16_t           *overflow
        ); """
    def_symbol(lib, "ps6000GetValuesOverlappedBulk", c_uint32,
               [c_int16, c_uint32, c_void_p, c_uint32, c_int32, c_uint32, c_uint32,
                c_void_p], doc)

    doc = """ PICO_STATUS ps6000GetValuesBulkAsyc
        (
            int16_t            handle,
            uint32_t           startIndex,
            uint32_t          *noOfSamples,
            uint32_t           downSampleRatio,
            PS6000_RATIO_MODE  downSampleRatioMode,
            uint32_t           fromSegmentIndex,
            uint32_t           toSegmentIndex,
            int16_t           *overflow
        ); """
    def_symbol(lib, "ps6000GetValuesAsync", c_uint32,
               [c_int16, c_uint32, c_uint32, c_uint32, c_int32, c_uint32, c_uint32,
                c_void_p], doc)

    doc = """ PICO_STATUS ps6000GetNoOfCaptures
        (
            int16_t   handle,
            uint32_t *nCaptures
        ); """
    def_symbol(lib, "ps6000GetNoOfCaptures", c_uint32, [c_int16, c_void_p], doc)

    doc = """ PICO_STATUS ps6000GetNoOfProcessedCaptures
        (
            int16_t   handle,
            uint32_t *nProcessedCaptures
        ); """
    def_symbol(lib, "ps6000GetNoOfProcessedCaptures", c_uint32, [c_int16, c_void_p],
               doc)

    """ PICO_STATUS ps6000Stop
        (
            int16_t  handle
        ); """
    def_symbol(lib, "ps6000Stop", c_uint32, [c_int16, ], doc)

    doc = """ PICO_STATUS ps6000SetNoOfCaptures
        (
            int16_t   handle,
            uint32_t  nCaptures
        ); """
    def_symbol(lib, "ps6000SetNoOfCaptures", c_uint32, [c_int16, c_uint32], doc)

    doc = """ PICO_STATUS ps6000SetWaveformLimiter
        (
            int16_t   handle,
            uint32_t  nWaveformsPerSecond
        ); """
    def_symbol(lib, "ps6000SetWaveformLimiter", c_uint32, [c_int16, c_uint32], doc)

    doc = """ PICO_STATUS ps6000EnumerateUnits
        (
            int16_t *count,
            int8_t  *serials,
            int16_t *serialLth
        ); """
    def_symbol(lib, "ps6000EnumerateUnits", c_uint32, [c_void_p, c_void_p, c_void_p],
               doc)

    doc = """ PICO_STATUS ps6000SetExternalClock
        (
            int16_t                    handle,
            PS6000_EXTERNAL_FREQUENCY  frequency,
            int16_t                    threshold
        ); """
    def_symbol(lib, "ps6000SetExternalClock", c_uint32, [c_int16, c_int32, c_int16],
               doc)

    doc = """ PICO_STATUS ps6000PingUnit
        (
            int16_t  handle
        ); """
    def_symbol(lib, "ps6000PingUnit", c_uint32, [c_int16, ], doc)

    doc = """ PICO_STATUS ps6000GetAnalogueOffset
        (
            int16_t          handle,
            PS6000_RANGE     range,
            PS6000_COUPLING  coupling,
            float           *maximumVoltage,
            float           *minimumVoltage
        ); """
    def_symbol(lib, "ps6000GetAnalogueOffset", c_uint32,
               [c_int16, c_int32, c_int32, c_void_p, c_void_p], doc)

    doc = """ PICO_STATUS ps6000GetTriggerInfoBulk
        (
            int16_t        handle,
            PS6000_TRIGGER_INFO    *triggerInfo,
            uint32_t      fromSegmentIndex,
            uint32_t      toSegmentIndex
        ); """
    def_symbol(lib, "ps6000GetTriggerInfoBulk", c_uint16,
               [c_int16, c_void_p, c_uint32, c_uint32], doc)
    # fmt: on

    return lib


_LIB_PS6000 = None


def load_ps6000():
    global _LIB_PS6000
    if _LIB_PS6000 is None:
        _LIB_PS6000 = _load_ps6000_lib()
        return _LIB_PS6000
    else:
        return _LIB_PS6000