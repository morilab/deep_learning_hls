<project xmlns="com.autoesl.autopilot.project" name="deep_learning_hls" top="deep_learning">
    <files>
        <file name="deep_learning_hls/Matrix.h" sc="0" tb="false" cflags=""/>
        <file name="deep_learning_hls/deep_learning.h" sc="0" tb="false" cflags=""/>
        <file name="../MNIST.cpp" sc="0" tb="1" cflags=""/>
        <file name="../MNIST.h" sc="0" tb="1" cflags=""/>
        <file name="../test.cpp" sc="0" tb="1" cflags=""/>
        <file name="../test.h" sc="0" tb="1" cflags=""/>
    </files>
    <includePaths/>
    <libraryPaths/>
    <Simulation argv="D:\\\\Projects\\\\FPGA\\\\deep_learning_hls\\\\MNIST\\\\">
        <SimFlow askAgain="false" name="csim" ldflags="" clean="true" csimMode="0" lastCsimMode="0"/>
    </Simulation>
    <solutions xmlns="">
        <solution name="solution1" status="active"/>
    </solutions>
</project>

